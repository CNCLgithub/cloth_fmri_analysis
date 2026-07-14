#!/bin/bash -e


if [ -z "${CFGFILE}" ]; then
    CFGFILE="default.conf" # change to "default.conf"
fi

. "$CFGFILE"

echo " "
if [[ $CFGFILE =~ event_* ]];then
    echo "==> Loading: "$CFGFILE
else
    echo "==> Loading: "$CFGFILE
    # export the required path variables
    for i in "${!PATHS[@]}"
    do
        # printf "%s \u2190 %s\n" "${i}" "${PATHS[$i]}"
        printf "export %s=\"%s\"\n" "${i}" "${PATHS[$i]}"
        export "${i}=${PATHS[$i]}"
    done
fi


#================== funcs =======================#
make_dir(){ if [ ! -d $1 ]; then mkdir -p $1; fi }
