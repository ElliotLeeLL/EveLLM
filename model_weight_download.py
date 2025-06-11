from utils.gpt_download import download_and_load_gpt2

if __name__ == '__main__':
    settings, params = download_and_load_gpt2(
        model_size="1558M", models_dir="gpt2"
    )
    print(settings)
    print(params.keys())