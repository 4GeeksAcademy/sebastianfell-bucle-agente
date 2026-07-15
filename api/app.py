"""
FastAPI REST API for inventory management.
Carla's coffee supply store - stock management system.
"""

import csv
import os
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INVENTORY_CSV = os.path.join(DATA_DIR, "inventory.csv")

app = FastAPI(
    title="Cafetería Inventory API",
    description="API para gestionar el inventario de suministros de cafetería",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ProductBase(BaseModel):
    name: str = Field(..., description="Nombre del producto")
    unit: str = Field(..., description="Unidad de medida (unidades, kg, litros, etc.)")
    min_stock: float = Field(
        ..., ge=0, description="Cantidad mínima antes de generar alerta"
    )
    category: str = Field(..., description="Categoría del producto")


class ProductCreate(ProductBase):
    quantity: float = Field(..., ge=0, description="Cantidad inicial en stock")


class ProductUpdate(BaseModel):
    quantity: float = Field(..., ge=0, description="Nueva cantidad en stock")
    reason: Optional[str] = Field(None, description="Motivo del cambio")


class ProductMove(BaseModel):
    delta: float = Field(..., description="Cambio de cantidad (positivo = entrada, negativo = salida)")
    reason: Optional[str] = Field(None, description="Motivo del cambio")


class Product(ProductBase):
    id: int
    quantity: float

    class Config:
        from_attributes = True


class Alert(BaseModel):
    product: Product
    deficit: float
    message: str


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _read_csv() -> List[dict]:
    """Read all rows from the inventory CSV and return them as a list of dicts."""
    if not os.path.exists(INVENTORY_CSV):
        return []
    with open(INVENTORY_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_csv(rows: List[dict]):
    """Overwrite the inventory CSV with the given rows."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(INVENTORY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row_to_product(row: dict) -> Product:
    """Convert a CSV row dict to a Product model."""
    return Product(
        id=int(row["id"]),
        name=row["name"],
        quantity=float(row["quantity"]),
        unit=row["unit"],
        min_stock=float(row["min_stock"]),
        category=row["category"],
    )


def _next_id(rows: List[dict]) -> int:
    """Return the next available product ID."""
    if not rows:
        return 1
    return max(int(r["id"]) for r in rows) + 1


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/products", response_model=List[Product])
def list_products(category: Optional[str] = None):
    """List all products, optionally filtered by category."""
    rows = _read_csv()
    products = [_row_to_product(r) for r in rows]
    if category:
        products = [p for p in products if p.category.lower() == category.lower()]
    return products


@app.get("/products/{product_id}", response_model=Product)
def get_product(product_id: int):
    """Get a single product by ID."""
    rows = _read_csv()
    for row in rows:
        if int(row["id"]) == product_id:
            return _row_to_product(row)
    raise HTTPException(status_code=404, detail="Producto no encontrado")


@app.post("/products", response_model=Product, status_code=201)
def create_product(product: ProductCreate):
    """Create a new product in the inventory."""
    rows = _read_csv()
    new_id = _next_id(rows)
    new_row = {
        "id": str(new_id),
        "name": product.name,
        "quantity": str(product.quantity),
        "unit": product.unit,
        "min_stock": str(product.min_stock),
        "category": product.category,
    }
    rows.append(new_row)
    _write_csv(rows)
    return _row_to_product(new_row)


@app.put("/products/{product_id}/quantity", response_model=Product)
def update_quantity(product_id: int, update: ProductUpdate):
    """Set the absolute quantity of a product."""
    rows = _read_csv()
    for i, row in enumerate(rows):
        if int(row["id"]) == product_id:
            rows[i]["quantity"] = str(update.quantity)
            _write_csv(rows)
            return _row_to_product(rows[i])
    raise HTTPException(status_code=404, detail="Producto no encontrado")


@app.patch("/products/{product_id}/move", response_model=Product)
def move_stock(product_id: int, move: ProductMove):
    """
    Adjust stock by a delta (positive = add stock, negative = remove stock).
    This is the natural endpoint for 'llegaron X unidades' or 'vendimos X'.
    """
    rows = _read_csv()
    for i, row in enumerate(rows):
        if int(row["id"]) == product_id:
            current = float(row["quantity"])
            new_qty = current + move.delta
            if new_qty < 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"No hay suficiente stock. "
                        f"Actual: {current} {row['unit']}, "
                        f"intentas quitar {abs(move.delta)}."
                    ),
                )
            rows[i]["quantity"] = str(new_qty)
            _write_csv(rows)
            return _row_to_product(rows[i])
    raise HTTPException(status_code=404, detail="Producto no encontrado")


@app.get("/alerts", response_model=List[Alert])
def get_alerts():
    """Return all products that are below their minimum stock level."""
    rows = _read_csv()
    alerts: List[Alert] = []
    for row in rows:
        product = _row_to_product(row)
        if product.quantity < product.min_stock:
            deficit = product.min_stock - product.quantity
            alerts.append(
                Alert(
                    product=product,
                    deficit=round(deficit, 2),
                    message=(
                        f"⚠️ {product.name}: solo quedan {product.quantity} {product.unit} "
                        f"(mínimo recomendado: {product.min_stock} {product.unit}). "
                        f"Te faltan {deficit:.2f} {product.unit} para cubrir la semana."
                    ),
                )
            )
    return alerts


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}