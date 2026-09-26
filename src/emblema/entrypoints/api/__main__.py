from emblema.config.settings import Settings
from emblema.entrypoints.api.api_server import ApiServer

if __name__ == "__main__":
    ApiServer(Settings()).run()
