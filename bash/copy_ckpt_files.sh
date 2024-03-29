#!/bin/bash
# Copies all ckpt files from Tensorflow Object Detection API training.

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/source/directory"
    exit 1
fi

src_dir="$1"

dst_dir="$src_dir/ckpt"

if [ ! -d "$dst_dir" ]; then
    mkdir -p "$dst_dir"
fi

while true; do
    for file in "$src_dir"/*; do
        if [[ "$file" == *ckpt-* && ! "$file" == *_temp* ]]; then
            base_filename=$(basename "$file")
            destination_file="$dst_dir/$base_filename"

	    if [ ! -f "$destination_file" ]; then
                cp "$file" "$dst_dir"
                # echo "File $file copied to $dst_dir"
	    fi
        fi
    done

    sleep 1
done
