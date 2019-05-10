from __future__ import print_function
from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
from apiclient.http import MediaFileUpload,MediaIoBaseDownload
import os
import io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n","--name", help="name of the file",type=str)
parser.add_argument("-id","--identify", help="id of the file",type=str)
parser.add_argument("-u","--upload", help="set to upload",action='store_true')
args = parser.parse_args()

# Setup the Drive v3 API
SCOPES = 'https://www.googleapis.com/auth/drive'
store = file.Storage(os.path.abspath('credentials.json'))
creds = store.get()
if not creds or creds.invalid:
    flow = client.flow_from_clientsecrets(os.path.abspath('client_secret_722001502329-'
                                          'obmekdep7i3as6cjat4ltdohiislc1si.apps.googleusercontent.com.json'), SCOPES)
    creds = tools.run_flow(flow, store)
drive_service = build('drive', 'v3', http=creds.authorize(Http()))

def uploadFile(name):
    file_metadata = {
    'name': args.name,
    'mimeType': '*/*'
    }
    media = MediaFileUpload(os.path.abspath(name),
                            mimetype='*/*',
                            resumable=True)
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print ('File ID: ' + file.get('id'))

if args.upload:
    uploadFile(args.name)
else:
    data = drive_service.files().get(fileId=args.identify).execute()
    file_id = args.identify
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(data['name'], 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))
