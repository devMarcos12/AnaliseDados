import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

diretorio_atual = os.getcwd()

chrome_options = Options()
prefs = {
    "download.default_directory": diretorio_atual,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True
}
chrome_options.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(options=chrome_options)

def DownloadFiles():
    try:
        driver.get('https://basedosdados.org/search')
        actions = ActionChains(driver)
        wait = WebDriverWait(driver, 15)

        time.sleep(4)
        actions.send_keys(Keys.ESCAPE).perform()
        time.sleep(2)

        SearchBar = driver.find_element(By.CSS_SELECTOR, "input[placeholder='Pesquisar dados']")
        SearchBar.send_keys("Campeonatos de Futebol")
        time.sleep(2)
        actions.send_keys(Keys.ENTER).perform()

        Container = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[@href='https://basedosdados.org/dataset/c861330e-bca2-474d-9073-bc70744a1b23']")))
        Container.click()

        time.sleep(3)
        abas_abertas = driver.window_handles
        driver.switch_to.window(abas_abertas[-1])

        try:
            botao_fechar_modal = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[text()='×']")))
            botao_fechar_modal.click()
        except Exception as e:
            actions.send_keys(Keys.ESCAPE).perform()
            print("Execute ESC to confirm close modal")

        time.sleep(3)


        print("Iniciando download da Série A...")
        driver.execute_script("window.scrollBy(0, 500);")
        time.sleep(5)
        DownloadButton = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Download')]")))
        DownloadButton.click()
        time.sleep(5)

        driver.execute_script("window.scrollBy(0, 1000);")
        time.sleep(2)
        DownloadTableButton = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Download da tabela')]")))
        DownloadTableButton.click()
        
        time.sleep(10) 
        print("Série A baixada!")

        # Initialize Copa do Brasil Download

        CopadoBrasil = wait.until(EC.element_to_be_clickable((By.XPATH, "//p[text()='Copa do Brasil']")))
        CopadoBrasil.click()
        
        time.sleep(5)


        DownloadTableButton2 = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Download da tabela')]")))
        DownloadTableButton2.click()

        time.sleep(10)
        print("Copa do Brasil baixada com sucesso!")

    finally:
        driver.quit()
        print(f"Arquivos salvos na pasta: {diretorio_atual}")

if __name__ == "__main__":
    DownloadFiles()