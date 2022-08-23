#!/bin/sh

FILES=$1
# AWS_ACCESS_KEY=$2
# AWS_SECRET_KEY=$3
# AWS_TOKEN=$4
ORIGIN="duo"
SAVE_DIR="/home/nfs/sdavis/duo_test/20220804_s3_script"
FILETYPE="json"
GROUPBY="user.name"
TIMESTAMP="isotimestamp"
APP="application.name"
CITY="access_device.location.city"
STATE="access_device.location.state"
COUNTRY="access_device.location.country"
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
