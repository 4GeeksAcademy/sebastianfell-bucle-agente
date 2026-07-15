"""
Sistema de gestión de inventario para Cafetería CarlaBot.

Arranque rápido:
  Terminal 1:  uvicorn api.app:app --reload
  Terminal 2:  python agent.py

Requisitos:
  - Crea un archivo .env con GROQ_API_KEY=tu_clave
  - pip install fastapi uvicorn openai python-dotenv requests
"""

if __name__ == "__main__":
    print("🐍 Sistema de inventario CarlaBot")
    print("   Terminal 1: uvicorn api.app:app --reload")
    print("   Terminal 2: python agent.py")