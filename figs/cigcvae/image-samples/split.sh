for DATASET in "cifar10" "ffhq256"
do
    rm -rf ${DATASET}
    mkdir ${DATASET}
    RESOLUTION=256
    if [[ "$DATASET" =  "cifar10" ]]; then
        RESOLUTION=32
    fi
    for FILE in raw/${DATASET}/*.jpg
    do
        FULL_NAME=$(basename $FILE)
        NAME=${FULL_NAME%.*}
        IMG_IDX=${NAME##*_}
        let CROP_Y=$RESOLUTION*4+10
        let CROP_X=$RESOLUTION*2+4
        convert $FILE -crop ${CROP_X}x${CROP_Y}+0+0 ${DATASET}/${IMG_IDX}_left.jpg
        convert $FILE -crop 5000x${CROP_Y}+${CROP_X}+0 ${DATASET}/${NAME}.jpg
    done
done