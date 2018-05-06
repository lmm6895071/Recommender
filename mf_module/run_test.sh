
if [ "$1" == "ml-1m" ]
then
	python ./run.py \
	-d ../data/movielens/preprocessed/movielens_1m/cf/v8/ \
	-a ../data/movielens/preprocessed/movielens_1m/ \
	-k $7 \
	-u $3 \
	-v $4 \
	-P $5 \
	-Q $6 \
	-F $2 
elif [ "$1" == "ml-10m" ]
then
	python ./run.py \
	-d ../data/movielens/preprocessed/movielens_10m/cf/v8/ \
	-a ../data/movielens/preprocessed/movielens_10m/ \
	-k $7 \
	-u $3 \
	-v $4 \
	-P $5 \
	-Q $6 \
	-F $2
elif [ "$1" == "aiv" ]
then
	python ./run.py \
	-d ../data/aiv/preprocessed/aiv/cf/v8 \
	-a ../data/aiv/preprocessed/aiv \
	-k $7 \
	-u $3 \
	-v $4 \
	-P $5 \
	-Q $6 \
	-F $2 
fi

date
