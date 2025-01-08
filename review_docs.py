from pathlib import Path
import sys
from rich.console import Console
from rich.markdown import Markdown

console = Console()

def display_doc(doc_path):
    """Muestra el contenido de un documento MDX como markdown formateado"""
    print(f"\n{'='*80}\n{doc_path.name}\n{'='*80}")
    
    # Leer el contenido y convertirlo a markdown
    content = doc_path.read_text()
    md = Markdown(content)
    
    # Mostrar con formato
    console.print(md)

def list_docs():
    """Lista todos los documentos MDX en la carpeta docs"""
    docs_dir = Path("docs")
    if not docs_dir.exists():
        print("Error: La carpeta 'docs' no existe")
        return []
    
    docs = list(docs_dir.glob("*.mdx"))
    if not docs:
        print("No se encontraron archivos MDX en la carpeta docs")
        return []
    
    print("\nDocumentos encontrados:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. docs/{doc.name}")
    return docs

def main():
    docs = list_docs()
    if not docs:
        return

    while True:
        print("\nOpciones:")
        print("1. Ver un documento específico")
        print("2. Ver todos los documentos")
        print("3. Salir")
        
        choice = input("\nElija una opción (1-3): ")
        
        if choice == "1":
            num = input(f"Ingrese el número del documento (1-{len(docs)}): ")
            try:
                index = int(num) - 1
                if 0 <= index < len(docs):
                    display_doc(docs[index])
                else:
                    print("Número inválido")
            except ValueError:
                print("Por favor ingrese un número válido")
        
        elif choice == "2":
            for doc in docs:
                display_doc(doc)
                input("\nPresione Enter para continuar...")
        
        elif choice == "3":
            break
        
        else:
            print("Opción inválida")

if __name__ == "__main__":
    main() 