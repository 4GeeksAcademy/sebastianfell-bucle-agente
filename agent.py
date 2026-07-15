"""
AI Agent for inventory management.
Connects to a Groq LLM and uses the inventory API as tools.
Runs in a loop: observe → think → act → update → repeat.
All steps are logged to conversation_log.csv.
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

API_BASE_URL = "http://localhost:8000"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"  # available on Groq free tier

LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "conversation_log.csv"
)

if not GROQ_API_KEY or GROQ_API_KEY == "tu_clave_aqui":
    print(
        "❌ ERROR: No has configurado tu GROQ_API_KEY en el archivo .env.\n"
        "   Abre el archivo .env y reemplaza 'tu_clave_aqui' con tu clave real.\n"
        "   Puedes obtener una clave gratuita en https://console.groq.com/keys"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_event(actor: str, message: str, tool_call: str = ""):
    """Append one row to the conversation log CSV."""
    timestamp = datetime.now(timezone.utc).isoformat()
    row = {
        "actor": actor,
        "message": message.replace("\n", " ").replace("\r", " "),
        "tool_call": tool_call,
        "timestamp": timestamp,
    }
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["actor", "message", "tool_call", "timestamp"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Tool definitions  (the "tools" that the LLM can choose to call)
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "listar_productos",
            "description": "Lista todos los productos del inventario. Puedes filtrar por categoría (ej: Café, Lácteos, Desechables, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Categoría para filtrar (opcional). Ej: Café, Lácteos, Azúcares, Desechables, Repostería, Especias, Snacks",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "obtener_producto",
            "description": "Obtiene los detalles de un producto específico por su nombre o ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "El ID numérico del producto",
                    }
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "registrar_entrada",
            "description": "Registra una entrada de stock (ej: 'llegaron 30 unidades de leche de avena'). Usa delta positivo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "El ID del producto",
                    },
                    "delta": {
                        "type": "number",
                        "description": "Cantidad que entra (siempre positiva). Ej: 30",
                    },
                },
                "required": ["product_id", "delta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "registrar_salida",
            "description": "Registra una salida/venta de stock (ej: 'vendimos 12 bolsas de arábica'). Usa delta negativo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "El ID del producto",
                    },
                    "delta": {
                        "type": "number",
                        "description": "Cantidad que sale (siempre positiva, el sistema la convertirá automáticamente). Ej: 12",
                    },
                },
                "required": ["product_id", "delta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crear_producto",
            "description": "Crea un nuevo producto en el inventario.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nombre del producto",
                    },
                    "quantity": {
                        "type": "number",
                        "description": "Cantidad inicial",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unidad de medida: unidades, kg, litros, paquetes, etc.",
                    },
                    "min_stock": {
                        "type": "number",
                        "description": "Cantidad mínima recomendada",
                    },
                    "category": {
                        "type": "string",
                        "description": "Categoría del producto",
                    },
                },
                "required": ["name", "quantity", "unit", "min_stock", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alertas_stock_bajo",
            "description": "Obtiene todos los productos que están por debajo de su nivel mínimo de stock. Útil para responder '¿qué productos están por agotarse?'",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations  (actual Python functions that call the API)
# ---------------------------------------------------------------------------


def _api_get(path: str) -> Dict:
    """Helper: GET request to the API."""
    resp = requests.get(f"{API_BASE_URL}{path}")
    resp.raise_for_status()
    return resp.json()


def _api_post(path: str, data: dict) -> Dict:
    """Helper: POST request to the API."""
    resp = requests.post(f"{API_BASE_URL}{path}", json=data)
    resp.raise_for_status()
    return resp.json()


def _api_patch(path: str, data: dict) -> Dict:
    """Helper: PATCH request to the API."""
    resp = requests.patch(f"{API_BASE_URL}{path}", json=data)
    resp.raise_for_status()
    return resp.json()


def _api_put(path: str, data: dict) -> Dict:
    """Helper: PUT request to the API."""
    resp = requests.put(f"{API_BASE_URL}{path}", json=data)
    resp.raise_for_status()
    return resp.json()


def tool_listar_productos(category: Optional[str] = None) -> str:
    """List products, optionally filtered by category."""
    path = "/products"
    if category:
        path += f"?category={category}"
    products = _api_get(path)
    if not products:
        return "No se encontraron productos."
    lines = ["📦 **Productos en inventario:**"]
    for p in products:
        lines.append(
            f"  🔹 **{p['name']}** (ID: {p['id']}) — {p['quantity']} {p['unit']} "
            f"| Mín: {p['min_stock']} {p['unit']} | Categoría: {p['category']}"
        )
    return "\n".join(lines)


def tool_obtener_producto(product_id: int) -> str:
    """Get a single product by ID."""
    product = _api_get(f"/products/{product_id}")
    return (
        f"📦 **{product['name']}** (ID: {product['id']})\n"
        f"   Cantidad: {product['quantity']} {product['unit']}\n"
        f"   Stock mínimo: {product['min_stock']} {product['unit']}\n"
        f"   Categoría: {product['category']}"
    )


def tool_registrar_entrada(product_id: int, delta: float) -> str:
    """Register stock entry (positive delta)."""
    result = _api_patch(
        f"/products/{product_id}/move", {"delta": abs(delta)}
    )
    return (
        f"✅ **Entrada registrada.**\n"
        f"   Producto: {result['name']}\n"
        f"   Nuevo stock: {result['quantity']} {result['unit']}\n"
        f"   Se añadieron {abs(delta)} {result['unit']}."
    )


def tool_registrar_salida(product_id: int, delta: float) -> str:
    """Register stock exit/ sale (negative delta)."""
    result = _api_patch(
        f"/products/{product_id}/move", {"delta": -abs(delta)}
    )
    return (
        f"✅ **Salida registrada.**\n"
        f"   Producto: {result['name']}\n"
        f"   Nuevo stock: {result['quantity']} {result['unit']}\n"
        f"   Se quitaron {abs(delta)} {result['unit']}."
    )


def tool_crear_producto(
    name: str, quantity: float, unit: str, min_stock: float, category: str
) -> str:
    """Create a new product."""
    result = _api_post(
        "/products",
        {
            "name": name,
            "quantity": quantity,
            "unit": unit,
            "min_stock": min_stock,
            "category": category,
        },
    )
    return (
        f"✅ **Producto creado:** {result['name']} (ID: {result['id']})\n"
        f"   Stock inicial: {result['quantity']} {result['unit']}\n"
        f"   Stock mínimo: {result['min_stock']} {result['unit']}\n"
        f"   Categoría: {result['category']}"
    )


def tool_alertas_stock_bajo() -> str:
    """Get low-stock alerts."""
    alerts = _api_get("/alerts")
    if not alerts:
        return "✅ **Todo en orden.** No hay productos con stock bajo en este momento."
    lines = ["⚠️ **Productos con stock bajo:**"]
    for a in alerts:
        lines.append(f"  🔴 {a['message']}")
    return "\n".join(lines)


# Map tool names to their implementations
TOOL_IMPLEMENTATIONS: Dict[str, Callable] = {
    "listar_productos": tool_listar_productos,
    "obtener_producto": tool_obtener_producto,
    "registrar_entrada": tool_registrar_entrada,
    "registrar_salida": tool_registrar_salida,
    "crear_producto": tool_crear_producto,
    "alertas_stock_bajo": tool_alertas_stock_bajo,
}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_agent():
    """Main agent loop: observe → think → act → update → repeat."""

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
    )

    # System prompt — sets the agent's persona
    system_prompt = (
        "Eres CarlaBot, un asistente amable y experto en gestión de inventario "
        "para una tienda de suministros de cafetería. Hablas en español de forma "
        "natural y cercana, como si fueras un compañero de trabajo.\n\n"
        "Tus funciones:\n"
        "1. Registrar entradas de stock ('llegaron 30 unidades de leche de avena')\n"
        "2. Registrar ventas o salidas ('vendimos 12 bolsas de arábica')\n"
        "3. Consultar productos y sus cantidades\n"
        "4. Avisar cuando algo está por agotarse\n"
        "5. Ayudar a buscar productos por nombre o categoría\n\n"
        "REGLAS IMPORTANTES:\n"
        "- Cuando el usuario mencione un producto pero no sepas su ID, primero "
        "usa listar_productos para buscar el producto por nombre.\n"
        "- Si listar_productos devuelve muchos productos y ves el que buscas, "
        "usa su ID para la operación.\n"
        "- No inventes IDs de productos. Siempre obtén el ID real de la API.\n"
        "- Cuando registres entradas o salidas, confirma el resultado con el usuario.\n"
        "- Si el usuario dice 'se agotó X' o 'no queda X', verifica el stock primero.\n"
        "- Responde siempre en español, de forma clara y amigable."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt}
    ]

    log_event("agent", "Sesión iniciada", "system")

    print("🧠 CarlaBot — Asistente de Inventario")
    print("   Escribe 'salir' o 'exit' para terminar.\n")

    while True:
        # ── Observe ──────────────────────────────────────────────────────
        user_input = input("👤 Tú: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("salir", "exit", "quit"):
            log_event("user", user_input, "")
            print("👋 ¡Hasta luego! Que tengas un buen día.")
            log_event("agent", "Sesión finalizada por el usuario.", "system")
            break

        log_event("user", user_input, "")
        messages.append({"role": "user", "content": user_input})

        # ── Loop: think → act → update ──────────────────────────────────
        while True:
            try:
                # ── Think ──
                log_event("agent", f"Enviando mensaje al LLM ({MODEL_NAME})...", "llm_call")
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,  # type: ignore
                    tools=TOOLS,
                    tool_choice="auto",
                )
            except Exception as e:
                error_msg = f"Error al conectar con el LLM: {e}"
                print(f"  ❌ {error_msg}")
                log_event("tool", error_msg, "llm_error")
                break

            choice = response.choices[0]
            message = choice.message

            # If the LLM wants to call a tool
            if message.tool_calls:
                messages.append(message)  # add assistant message with tool calls

                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    log_event(
                        "agent",
                        f"Llamando a la herramienta '{fn_name}' con args: {fn_args}",
                        fn_name,
                    )
                    print(f"  🤖 Llamando a: {fn_name}({fn_args})...")

                    # ── Act ──
                    try:
                        impl = TOOL_IMPLEMENTATIONS.get(fn_name)
                        if impl:
                            result_str = impl(**fn_args)
                        else:
                            result_str = f"Error: herramienta '{fn_name}' no implementada."
                    except requests.exceptions.ConnectionError:
                        result_str = (
                            "❌ ERROR: No se pudo conectar con la API. "
                            "¿Está el servidor de la API corriendo? "
                            "Ejecuta 'uvicorn api.app:app --reload' en otro terminal."
                        )
                        print(f"  {result_str}")
                    except Exception as e:
                        result_str = f"❌ Error al ejecutar {fn_name}: {e}"
                        print(f"  {result_str}")

                    log_event("tool", result_str, fn_name)

                    # ── Update ──
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str,
                        }
                    )

                # After processing tools, loop back to let LLM respond
                continue

            # ── Final response ────────────────────────────────────────────
            if message.content:
                print(f"  🤖 CarlaBot: {message.content}")
                log_event("agent", message.content, "")
                messages.append({"role": "assistant", "content": message.content})
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_agent()
    except KeyboardInterrupt:
        print("\n👋 Interrupción recibida. ¡Hasta luego!")
        log_event("agent", "Sesión interrumpida por el usuario (Ctrl+C).", "system")
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        log_event("agent", f"Error inesperado: {e}", "system")
        sys.exit(1)