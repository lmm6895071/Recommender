date 
if [ "$1" == "ml-1m" ]
then
	python ./run.py \
	-d ../data/movielens/preprocessed/movielens_1m/cf/v$2/ \
	-a ../data/movielens/preprocessed/movielens_1m/ \
	-c True \
	-r ../data/movielens/ml-1m_ratings.dat \
	-i ../data/movielens/ml_plot.dat \
	-m 20 \
	-t $3
elif [ "$1" == "ml-10m" ]
then
	python ./run.py \
	-d ../data/movielens/preprocessed/movielens_10m/cf/v$2/ \
	-a ../data/movielens/preprocessed/movielens_10m/ \
	-c True \
	-r ../data/movielens/ml-10m_ratings.dat \
	-i ../data/movielens/ml_plot.dat \
	-m 1 \
	-t $3
elif [ "$1" == "aiv" ]
then
	python ./run.py \
	-d ../data/aiv/preprocessed/aiv/cf/v$2 \
	-a ../data/aiv/preprocessed/aiv \
	-c True \
	-r ../data/aiv/Amazon_Instant_Video_ratings.txt \
	-i ../data/aiv/Amazon_Instant_Video_items.txt \
	-m 3 \
	-t $3 \
	-l 300
elif [ "$1" == "IMDB" ]
then
	echo "IMDB"
elif [ "$1" == "yelp" ]
then
	python ./run.py \
	-d ../data/yelp/yelp$4/preprocessed/yelp$4/cf/v$2 \
	-a ../data/yelp/yelp$4/preprocessed/yelp$4/ \
	-c True \
	-r ../data/yelp/yelp14/ratings.txt \
	-i ../data/yelp/yelp14/items.txt \
	-s 45000 \
	-l 1000 \
	-m 1 \
	-t $3
fi
date
