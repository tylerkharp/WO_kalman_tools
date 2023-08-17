# Copyright (c) 2023 Gecko Robotics, Inc. All rights reserved.
# Authored by Ryan Dickerhoff - ryan.dickerhoff@geckorobotics.com
from google.auth.transport.requests import Request # pip install google-auth
from google.oauth2 import id_token # pip install google-auth
import requests # pip install requests
import json
import os
import pandas as pd # pip install pandas
import logging
import gzip

GOOGLE_CLIENT_ID = "489098902706-dmprf9aadrnbhlvqvmqbhidac6v3ir2l.apps.googleusercontent.com"
LOGFILE_URL_BASE = "https://terrarium.internal.geckorobotics.com/tracking/api/logfile"

def set_google_application_credentials(authentication_directory) -> None:
    #list files in the path os.path.join(os.getcwd(), "auth")
    # path = os.path.join(, "auth")
    auth_file = os.listdir(authentication_directory)
    if len(auth_file) == 1:
        auth_path = os.path.join(authentication_directory, auth_file[0])
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = auth_path
        logging.info("GOOGLE_APPLICATION_CREDENTIALS set to " + auth_path)
    else:
        logging.warning("Could not find the auth file.\n\tPlease ensure that the auth file is the only file present in the auth folder.\n\tIf you dont have an auth file yet, create one here: https://console.cloud.google.com/iam-admin/serviceaccounts/details/108618847850579867781/keys?authuser=0&project=gecko-admin&supportedpurview=project")  

def get_log_filenames_from_slug(slug:str,slug_info:dict) -> pd.DataFrame:
    """ Retrieves log filenames associated with the given list of slugs and returns them as a DataFrame.

    This function takes one slug file name, fetches the associated log filenames using an internal API,
    and returns the log filenames as a pandas DataFrame. It also logs a warning when no log files are found
    for the slug.

    Args:
        slug (str): A singular slug name for which log filenames are to be fetched.

    Returns:
        pd.DataFrame: A DataFrame with keys as slugs and their respective log filenames as values,
                    after removing empty columns.
    """
    slug_log_files = {slug:[]}
    slug_event_files = {slug:[]}

    log_files_url_base=LOGFILE_URL_BASE + "/?slug="
    response = make_iap_request(log_files_url_base+slug, GOOGLE_CLIENT_ID)
    file_responses = json.loads(response)["results"]

    if not file_responses:
        start_date_query = f"timestamp__gte={slug_info['start_date']}T00:00:00-00:00"
        stop_date_query = f"timestamp__lte={slug_info['end_date']}T23:59:59-00:00"
        for computer in slug_info['computer_list']:
            log_files_url_base = LOGFILE_URL_BASE + "/?"
            computer_query = f"filename={computer}"
            log_files_url_base = log_files_url_base + computer_query + "&" + start_date_query + "&" + stop_date_query
            response = make_iap_request(log_files_url_base, GOOGLE_CLIENT_ID)
            file_responses = json.loads(response)["results"]
            log_files = [response["filename"] for response in file_responses]
            event_files = [response["events_filename"] for response in file_responses]
            slug_log_files[slug] += log_files
            slug_event_files[slug] += event_files
    else:
        slug_log_files[slug] = [response["filename"] for response in file_responses]
        slug_event_files[slug] = [response["events_filename"] for response in file_responses]
    
    return slug_log_files[slug], slug_event_files[slug]

def get_log_filenames_from_slug_list(slug_list:list) -> pd.DataFrame:
    """ Retrieves log filenames associated with the given list of slugs and returns them as a DataFrame.

    This function takes a list of slug_files, fetches the associated log filenames using an internal API,
    and returns the log filenames as a pandas DataFrame. It also logs a warning when no log files are found
    for a particular slug and raises an exception if no log files are found for any of the provided slugs.

    Args:
        slug_list (list): A list of slugs for which log filenames are to be fetched.

    Returns:
        pd.DataFrame: A DataFrame with keys as slugs and their respective log filenames as values,
                    after removing empty columns.
    """
    slug_files_df = pd.DataFrame(columns=['slug','log file names', 'event file names'])
    for slug in slug_list:
        log_files_for_slug, event_files_for_slug = get_log_filenames_from_slug(slug)
        slug_files_df_row = {'slug':slug, 'log file names':log_files_for_slug, 'event file names':event_files_for_slug}
        slug_files_df = slug_files_df.append(slug_files_df_row, ignore_index=True)

    if not slug_files_df['log file names'].any():
        raise Exception("no logfiles found for the provided slug")  
    
    remove_empty_columns(slug_files_df)
    return slug_files_df


def get_log_download_link(log_filename:str) -> str:
    """ Retrieves the download link for the specified log filename.

    This function takes the log_filename as input, constructs the request URL using
    an internal API, and makes an Identity-Aware Proxy-protected request to fetch the
    download link for the specified log file.

    Args:
        log_filename (str): The log filename for which the download link is to be fetched.

    Returns:
        str: The download link for the specified log filename.
    """
    url = LOGFILE_URL_BASE + "/download/csv/" + log_filename
    response = make_iap_request(url, GOOGLE_CLIENT_ID)
    link = json.loads(response)["fileUrl"]
    return link

def get_events_download_link(events_filename:str) -> str:
    """ Retrieves the download link for the specified events filename.

    This function takes the events_filename as input, constructs the request URL using
    an internal API, and makes an Identity-Aware Proxy-protected request to fetch the
    download link for the specified events file.

    Args:
        events_filename (str): The events filename for which the download link is to be fetched.

    Returns:
        str: The download link for the specified events filename.
    """
    url = LOGFILE_URL_BASE + "/download/events/" + events_filename
    response = make_iap_request(url, GOOGLE_CLIENT_ID)
    link = json.loads(response)["fileUrl"]
    return link

def download_gcs_file(link: str) -> str: 
    """ Download a file from a Google Cloud Storage link.

    This function downloads the content of a file from a Google Cloud Storage
    link using a GET request and returns the content as a string.

    Args:
        link (str): The Google Cloud Storage link of the file to be downloaded.

    Returns:
        str: The content of the downloaded file as a string.

    Example:
        content = download_gcs_file("https://storage.googleapis.com/my-bucket/my-file.txt")
    """
    if link is not None:
        response = requests.get(link)
        return response.text
    else:
        print("File link does not exist")


def make_iap_request(url, client_id, method='GET', **kwargs) -> str:
    """Makes a request to an application protected by Identity-Aware Proxy.

    Args:
      url: The Identity-Aware Proxy-protected URL to fetch.
      client_id: The client ID used by Identity-Aware Proxy.
      method: The request method to use
              ('GET', 'OPTIONS', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE')
      **kwargs: Any of the parameters defined for the request function:
                https://github.com/requests/requests/blob/master/requests/api.py
                If no timeout is provided, it is set to 90 by default.

    Returns:
      The page body, or raises an exception if the page couldn't be retrieved.

    Authored by Google TODO: include reference to licence

    """
    # Set the default timeout, if missing
    if 'timeout' not in kwargs:
        kwargs['timeout'] = 240

    # Obtain an OpenID Connect (OIDC) token from metadata server or using service
    # account.
    open_id_connect_token = id_token.fetch_id_token(Request(), client_id)

    # Fetch the Identity-Aware Proxy-protected URL, including an
    # Authorization header containing "Bearer " followed by a
    # Google-issued OpenID Connect token for the service account.
    resp = requests.request(
        method, url,
        headers={'Authorization': 'Bearer {}'.format(
            open_id_connect_token)}, **kwargs)
    if resp.status_code == 403:
        raise Exception('Service account does not have permission to '
                        'access the IAP-protected application.')
    elif resp.status_code != 200:
        raise Exception(
            'Bad response from application: {!r} / {!r} / {!r}'.format(
                resp.status_code, resp.headers, resp.text))
    else:
        return resp.text
    
def remove_empty_columns(df:pd.DataFrame) -> pd.DataFrame:
    """ Remove empty columns from a given DataFrame.

    This function removes columns containing only NaN values from a given
    Pandas DataFrame and returns the resulting DataFrame with empty columns removed.

    Args:
        df (pd.DataFrame): The input DataFrame from which to remove the empty columns.

    Returns:
        pd.DataFrame: The resulting DataFrame with empty columns removed.

    Example:
        cleaned_df = remove_empty_columns(input_dataframe)
    """
    df = df.dropna(axis=1, how='all')
    return df


def get_slug_file_content_from_terrarium(log_file_list,event_file_list) -> dict:
    """ gets the content of terrarium files for a slug

    This function gets all the log files' and event files' content associated with a certain slug

    Args:
        log_file_list: a List of the log file names
        event_file_list: a List of the event file names

    Returns:
        dict: with the slug name as the key, and the values as the contents of the files names given as inputs."""
    print("Starting to fetch files from terrarium...")
    log_file_content_list = {}
    event_file_content_list = {}
    count = 0
    for file_name in log_file_list:   
        log_download_link = get_log_download_link(file_name)
        log_file_content = download_gcs_file(log_download_link)
        log_file_content_list.update({file_name:log_file_content})
        print(f"Log File {count+1} of {len(log_file_list)} fetched ({file_name})")
        count+=1
    count = 0
    for file_name in event_file_list:
        events_download_link = get_events_download_link(file_name)
        event_file_content = download_gcs_file(events_download_link)
        event_file_content_list.update({file_name:event_file_content})
        print(f"Event File {count+1} of {len(log_file_list)} fetched")
        count+=1
    return log_file_content_list,event_file_content_list

def get_slug_files_from_terrarium(slug,slug_info,authentication_directory) -> pd.DataFrame:
    """get all the file content associated with a slug name from terrarium

    Args:
        slug (str): a name of a slug
        authentication_directory (str): the path to where you Google Cloud JSON key is stored

    Returns:
        log_file_content (list): list of strings of all content in the log file
        event_file_content (list): list of strings of all content in the event files."""
    
    set_google_application_credentials(authentication_directory)
    log_file_list, event_file_list = get_log_filenames_from_slug(slug,slug_info)
    log_file_content_list,event_file_content_list = get_slug_file_content_from_terrarium(log_file_list,event_file_list)
    return log_file_content_list,event_file_content_list     



def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("Testing terrarium utils")
    logging.debug("setting google application credentials")
    set_google_application_credentials(authentication_directory='/home/tyler.harp/UTScopeData/authentication')

    logging.debug("\nTesting get_log_filenames_from_slugs")
    log_and_event_files_df = get_log_filenames_from_slug_list(["20230509-4aeb47","20210802-827ff7"])
    logging.debug(log_and_event_files_df)
    first_logfile_name = log_and_event_files_df.iat[0,0]
    logging.debug(first_logfile_name)

    logging.debug("Testing saving files to data folder")
    log_content_list,event_content_list = get_slug_file_content_from_terrarium(log_and_event_files_df.iloc[0,1],log_and_event_files_df.iloc[0,2])
    terrarium_folder_path = '/home/tyler.harp/TerrariumData' #make this directory wherever you want to tsave the files
    for key in log_content_list:
        name_index = key.rfind('/')
        shortened_key = key[name_index+1:]
        with open (terrarium_folder_path + '/' + f"{shortened_key}.csv", "w") as f:
            f.write(log_content_list[key])
    for key in event_content_list:
        name_index = key.rfind('/')
        shortened_key = key[name_index+1:]
        with open (terrarium_folder_path + '/' + f"{shortened_key}", "w") as f:
            f.write(event_content_list[key])

if __name__ == "__main__":
    main()
