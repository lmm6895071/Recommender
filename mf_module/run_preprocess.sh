date 
if [ "$1" == "ml-1m" ]
then
	python ./run.py \
	-d ../data/movielens/preprocessed/movielens_1m/cf/v$2/ \
	-a ../data/movielens/preprocessed/movielens_1m/ \
	-c True \
	-r ../data/movielens/ml-1m_ratings.dat \
	-m 1 \
	-t $3
elif [ "$1" == "ml-10m" ]
then
	python ./run.py \
	-d ../data/movielens/preprocessed/movielens_10m/cf/v$2/ \
	-a ../data/movielens/preprocessed/movielens_10m/ \
	-c True \
	-r ../data/movielens/ml-10m_ratings.dat \
	-m 1 \
	-t $3
elif [ "$1" == "aiv" ]
then
	python ./run.py \
	-d ../data/aiv/preprocessed/aiv/cf/v$2 \
	-a ../data/aiv/preprocessed/aiv \
	-c True \
	-r ../data/aiv/Amazon_Instant_Video_ratings.txt \
	-m 3 \
	-t $3 \
	-l 300
fi

date