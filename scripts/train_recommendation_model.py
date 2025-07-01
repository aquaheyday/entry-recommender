from core.train_user import train_models_for_site

if __name__ == "__main__":
    site = input("Site ID to train: ")
    result = train_models_for_site(site)
    print("Trained:", result)
