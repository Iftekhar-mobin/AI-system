import pandas as pd
from os import path
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from amie_core.core.constants import DEFAULT_CORPUS

options = webdriver.ChromeOptions()
options.add_argument("headless")
options.add_argument('--start-maximized')
driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)
driver.set_window_size(1500, 3000)


def get_links(response, data_file):
    collector = []
    for items in response['page_relevance']:
        collector.append(items['doc_id'])
    return collector, urls(collector, data_file)


def urls(collector, data_file):
    dataset = pd.read_csv(data_file)
    saver = []
    for items in collector:
        saver.append(list(set(dataset[dataset.page == items].url))[0])
    return saver


class ScreenShotTaker:
    def __init__(self, data_dir, response):
        self.ids, self.links = get_links(response, path.join(data_dir, DEFAULT_CORPUS))
        self.resource_dir = data_dir
        self.take_screenshot()

    def take_screenshot(self):
        i = 0
        for link in self.links:
            driver.get(link)
            f_name = "Page_" + str(self.ids[i]) + ".png"
            driver.save_screenshot(path.join(self.resource_dir, f_name))
            i += 1
        driver.close()
