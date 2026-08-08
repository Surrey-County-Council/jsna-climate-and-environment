

def refresh_all_data():
    from jsna_climate_and_environment.datasets import (
        access_to_nature,
        ahah,
        met_office
    )
    access_to_nature.refresh_data()
    ahah.refresh_data()
    met_office.refresh_data()


if __name__ == "__main__":
    refresh_all_data()
