
if [ "$1" == "ml-1m" ]
then
	CUDA_VISIBLE_DEVICES="$8"  python ./run.py \
	-d ../data/movielens/preprocessed/movielens_1m/cf/v8/ \
	-a ../data/movielens/preprocessed/movielens_1m/ \
	-o ../result/movielens_1m/result/1_100_200 \
	-e 200 \
	-k $7 \
	-p ../data/preprocessed/glove.6B.200d.txt \
	-u $3 \
	-v $4 \
	-P $5 \
	-Q $6 \
	-g True \
	-F $2 

elif [ "$1" == "ml-10m" ]
then
	CUDA_VISIBLE_DEVICES="$8" python ./run.py \
	-d ../data/movielens/preprocessed/movielens_10m/cf/v8/ \
	-a ../data/movielens/preprocessed/movielens_10m/ \
	-o ../result/movielens_10m/result/1_100_200 \
	-e 200 \
	-k $7 \
	-p ../data/preprocessed/glove.6B.200d.txt \
	-u $3 \
	-v $4 \
	-g True \
	-P $5 \
	-Q $6 \
	-F $2
elif [ "$1" == "aiv" ]
then
	CUDA_VISIBLE_DEVICES="$8" python ./run.py \
	-d ./data/aiv/preprocessed/aiv/cf/v8 \
	-a ./data/aiv/preprocessed/aiv \
	-o ./result/aiv/result/  \
	-e 200 \
	-l 300 \
	-k $7 \
	-p ./data/preprocessed/glove.6B.200d.txt \
	-u $3 \
	-v $4 \
	-P $5 \
	-Q $6 \
	-g True \
	-F $2

elif [ "$1" == "IMDB" ]
then
	CUDA_VISIBLE_DEVICES="$8"  python ./run.py \
	-d ../data/imdb/preprocessed/imdb/cf/v8.0/ \
	-a ../data/imdb/preprocessed/imdb/ \
	-o ./result/imdb/result/1_100_200 \
	-e 200 \
	-k $7 \
	-p ../data/preprocessed/glove.6B.200d.txt \
	-u $3 \
	-v $4 \
	-P $5 \
	-Q $6 \
	-g True \
	-F $2 
elif [ "$1" == "yelp" ]
then
	CUDA_VISIBLE_DEVICES="$9"  python ./run.py \
	-d ../data/yelp/yelp$8/preprocessed/yelp$8/cf/v8.0/ \
	-a ../data/yelp/yelp$8/preprocessed/yelp$8/ \
	-o ../result/yelp$6/result/1_100_200 \
	-e 200 \
	-k $7 \
	-w 100 \
	-s 45000 \
	-l 1000 \
	-p ../data/preprocessed/embeding_yelp$8 \
	-u $3 \
	-v $4 \
	-P $5 \
	-Q $6 \
	-g True \
	-F $2 

fi
date
