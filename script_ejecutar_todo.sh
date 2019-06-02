
echo "" > resultados.txt

echo "|=========================|" >> resultados.txt
echo "|-------Imagen 720px------|" >> resultados.txt
echo "|=========================|" >> resultados.txt

for kernel in 2 3 4 5 6 7 8 9 10 11 12 13 14;
do
    echo "-------Kernel: $kernel -----" >> resultados.txt
    { time ./blur-effect 720-image.jpg ./outputs/720-image-out-$kernel.jpg $kernel >/dev/null 2>&1;} |&  tee -a resultados.txt
done


echo "|=========================|" >> resultados.txt
echo "|-------Imagen 1080px-----|" >> resultados.txt
echo "|=========================|" >> resultados.txt

for kernel in 2 3 4 5 6 7 8 9 10 11 12 13 14;
do
    echo "-------Kernel: $kernel ------" >> resultados.txt
    { time ./blur-effect 1080-image.jpeg ./outputs/1080-image-out-$kernel.jpeg $kernel>/dev/null 2>&1;} |&  tee -a resultados.txt
done
echo "|=========================|" >> resultados.txt
echo "|--------Imagen 4K--------|" >> resultados.txt
echo "|=========================|" >> resultados.txt

for kernel in 2 3 4 5 6 7 8 9 10 11 12 13 14;
do
    echo "-------Kernel: $kernel ------" >> resultados.txt
    { time ./blur-effect 4k-image.jpg ./outputs/4k-image-out.jpg-$kernel $kernel >/dev/null 2>&1;} |&  tee -a resultados.txt
done

