from shutil import which

SELENIUM_DRIVER_NAME = 'chrome'
SELENIUM_DRIVER_EXECUTABLE_PATH = which('chromedriver')
SELENIUM_DRIVER_ARGUMENTS = 'headless'

DOWNLOAD_MIDDLEWARES = {
    'scrapy_selenium.SeleniumMiddleware': 800
}
