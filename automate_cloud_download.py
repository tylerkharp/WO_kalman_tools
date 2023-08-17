import os

def get_slug(slug):
    command = f"SLUG={slug};gsutil -m cp -n -r gs://gecko-archive/$SLUG/*.json gs://gecko-archive/$SLUG/config gs://gecko-archive/$SLUG/*_ut.csv gs://gecko-signals-warm/$SLUG/*.bin* gs://gecko-archive/$SLUG/*_localization.csv /media/tyler.harp/gecko_data/cloud_data/{slug}"
    os.system(command)
def download_all_slugs(slug_list):
    for slug in slug_list:
        get_slug(slug)
def make_slug_folders(slug_list):
    for slug in slug_list:
        if not os.path.exists(f"/media/tyler.harp/gecko_data/cloud_data/{slug}"):
            os.mkdir(f"/media/tyler.harp/gecko_data/cloud_data/{slug}")

if __name__ == "__main__":
    slug_list = [
                "20230508-04487e",]
    make_slug_folders(slug_list)
    download_all_slugs(slug_list)