from playwright.sync_api import sync_playwright, expect
import os

def run_verification(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()

    try:
        # 1. Navigate to the app
        page.goto("http://localhost:8501")

        # 2. Wait for the main header to be visible, ensuring the app is loaded.
        # Use a generous timeout as the app might take a moment to start.
        header_locator = page.get_by_role("heading", name="EDA Agente Inteligente")
        expect(header_locator).to_be_visible(timeout=30000)

        # 3. Select the example dataset
        # The app uses example URLs, so we'll select the Titanic dataset
        page.get_by_role("tab", name="Exemplos").click()
        page.get_by_role("combobox").select_option(label="Titanic Dataset (histórico)")

        # 4. Click the button to process the dataset
        page.get_by_role("button", name="Processar Carregamento").click()

        # 5. Wait for the chat interface to appear after loading
        expect(page.get_by_text("Chat com Agentes Inteligentes")).to_be_visible(timeout=60000)

        # 6. Ask for a visualization
        question_area = page.get_by_role("textbox", name="Digite sua pergunta sobre os dados:")
        question_area.fill("cria matriz de correlação")
        page.get_by_role("button", name="Enviar").click()

        # 7. Wait for the visualization to be rendered
        # The visualization is inside a canvas element in a div with role="img"
        expect(page.get_by_text("Visualização Gerada")).to_be_visible(timeout=60000)

        # 8. Take a screenshot
        screenshot_path = "jules-scratch/verification/verification.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        page.screenshot(path="jules-scratch/verification/error.png")

    finally:
        browser.close()

with sync_playwright() as playwright:
    run_verification(playwright)