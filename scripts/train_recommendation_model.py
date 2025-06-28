from core.train_user import train_model_for_site

if __name__ == "__main__":
    site = input("Site ID to train: ")
    result = train_model_for_site(site)
    print("Trained:", result)
