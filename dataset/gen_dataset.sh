while read in; do
    echo Processing "$in"

    bn=$(basename "$in")
    outdir=output/$bn

    mkdir -p "$outdir"

    ffmpeg -nostdin -i "$in" -vf "select=not(mod(n\,30)), scale=-1:256, crop=256:256" -vsync vfr -q:v 2 "$outdir/%07d.jpg"

done <video_path.in
