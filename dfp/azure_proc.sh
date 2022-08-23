#!/bin/sh

FILES=$1
ORIGIN="azure"
SAVE_DIR="/home/nfs/sdavis/azure_test/20220804_s3_script"
FILETYPE="json"
GROUPBY="properties.userPrincipalName"
TIMESTAMP="properties.createdDateTime"
APP="properties.appDisplayName"
CITY="properties.location.city"
STATE="properties.location.state"
COUNTRY="properties.location.countryOrRegion"
EXTENSION=".json"
MIN_RECORDS=0

python preprocess.py --origin $ORIGIN \
 --files $FILES \
 --save_dir $SAVE_DIR \
 --filetype $FILETYPE \
 --groupby $GROUPBY \
 --timestamp $TIMESTAMP \
 --city $CITY \
 --state $STATE \
 --country $COUNTRY \
 --app $APP \
 --extension $EXTENSION \
 --min_records $MIN_RECORDS
