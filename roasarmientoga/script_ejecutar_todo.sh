
echo "" > resultados.txt

echo "--------Imagen 720px--------" >> resultados.txt
for hilo in 1 2 4 8 16;
    do
        for kernel in 4 6 8 10 12 14;
        do
            echo "-------Kernel: $kernel ------ No de Hilos: $hilo ---------" >> resultados.txt
            { time ./blur-effect 720-image.jpg 720-image-out.jpg $kernel $hilo >/dev/null 2>&1;} |&  tee -a resultados.txt
        done
    done
echo "--------Imagen 1080px--------" >> resultados.txt
for hilo in 1 2 4 8 16;
    do
        for kernel in 4 6 8 10 12 14;
        do
            echo "-------Kernel: $kernel ------ No de Hilos: $hilo ---------" >> resultados.txt
            { time ./blur-effect 1080-image.jpeg 1080-image-out.jpeg $kernel $hilo >/dev/null 2>&1;} |&  tee -a resultados.txt
        done
    done
echo "--------Imagen 4K--------" >> resultados.txt
for hilo in 1 2 4 8 16;
    do
        for kernel in 4 6 8 10 12 14;
        do
            echo "-------Kernel: $kernel ------ No de Hilos: $hilo ---------" >> resultados.txt
            { time ./blur-effect 4k-image.jpg 4k-image-out.jpg $kernel $hilo >/dev/null 2>&1;} |&  tee -a resultados.txt
        done
    done

