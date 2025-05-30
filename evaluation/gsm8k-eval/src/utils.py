import os


def construct_folder(folders):
    for folder in folders:
        if not os.path.isdir(folder):
            os.makedirs(folder)


def check_construct(folders):
    for folder in folders:
        if not os.path.isdir(folder):
            return False
    return True


def init_env_file(
    env_path=".env",
    default_content='OPENAI_API_KEY="1234567 (replace with your own)"\nBASE_URL="example.chat (replace with your own)"',
):
    if not os.path.exists(env_path):
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(default_content)
        print("WARNING, remember to use your own api secret key")


def split_print():
    print("===============================================")


def load_api_key():
    # init api file
    init_env_file()

    import dotenv

    dotenv.load_dotenv(override=True)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("BASE_URL")
    # print(OPENAI_API_KEY)
    # print(BASE_URL)
    return OPENAI_API_KEY, BASE_URL


if __name__ == "__main__":
    # load_api_key()
    # init_env_file()
    # print(load_api_key())
    pass
