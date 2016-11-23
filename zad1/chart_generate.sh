 #!/bin/bash
#Ver v2

echo "Skrypt do generowania wykresu zależności czasu wykonania oraz przyspieszenia od liczby wątków wykorzystując gnuplot'a"


#Gdy brak parametrow, foch und brexit
if [ $# -ne 2 ] 
	then
		printf "\nPodajże parametry, huncwocie!\n\n"
		printf "Pierwszy parametr to nazwa programu, drugi to rozmiar danych wejściowych\n"
		printf "Przykładowe uruchomienie: $0 macierz_omp 100000\n\n"	
		exit
fi

GNUPLOT_CHART_CONFIG_FILENAME="gnuplot_charts.config"

PROGRAM_NAME=$1
DATA_SIZE=$2

TIME_CHART_FILENAME="wykres_czasu"
SPEEDUP_CHART_FILENAME="wykres_przyspieszenia"

MAX_REPEAT_FOR_AVG=10
MAX_THREADS=15

# tworzymy nowy plik z danymi do wykresu:
> $TIME_CHART_FILENAME.data
> $SPEEDUP_CHART_FILENAME.data

ONE_THREAD_SPEED=$((0))
echo "Czasy obliczeń dla liczby $DATA_SIZE danych"

for ((i=1; i<=$MAX_THREADS; i++)) 
do

	AVG_MS_TIME=$((0))
	for ((k=1; k<=$MAX_REPEAT_FOR_AVG; k++))
	do
		OUTPUT="$(./$PROGRAM_NAME $i $DATA_SIZE)"
		echo $OUTPUT
		# Tu poniżej zbieramy tylko liczby z wyjścia naszego programu. 
		# Dlatego ważne żeby nie było np liczbowych wyników pośrednich
		CZAS_W_MS=$(echo $OUTPUT | tr -dc '0-9')
		AVG_MS_TIME=$((AVG_MS_TIME + CZAS_W_MS))
	done
	
	AVG_MS_TIME=$((AVG_MS_TIME / 10))
	echo -n "Średni czas uruchomienia dla $i wątków to: $AVG_MS_TIME ms"
	echo $i $AVG_MS_TIME >> $TIME_CHART_FILENAME.data
	
	if [ $i -eq 1 ]
		then
			ONE_THREAD_SPEED=$AVG_MS_TIME
	fi

	AVG_SPEEDUP=$( echo  "$ONE_THREAD_SPEED / $AVG_MS_TIME" | bc -l)
	echo ", przyśpieszenie: $AVG_SPEEDUP"
	echo $i $AVG_SPEEDUP >> $SPEEDUP_CHART_FILENAME.data

done
