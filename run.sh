docker build -t myimage .
# docker run -p 8789:8789 -d myimage -v /Users/terng/Downloads/celebrity-face-recognition:/code/dataset
# docker run -p 8789:8789 -d myimage --mount spwdrc=/Users/terng/Downloads/celebrity-face-recognition,target=/code/dataset,type=bind 
# docker run -v $(pwd)/celebrity-face-recognition:/code/dataset -p 8789:8789 -d myimage
docker run --volume=/Users/terng/Downloads/celebrity-face-recognition:/code/dataset -p 8789:8789 -d myimage
# docker run --volume=path/to/celebrity-face-recognition:/code/dataset -p 8789:8789 -d myimage
