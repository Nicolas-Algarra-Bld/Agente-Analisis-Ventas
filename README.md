Para ejecutar el programa se debe asegurar tener las credenciales necesarias en un archivo .env siguiendo la estructura del archivo .env.example del proyecto. 
Luego es necesario ejecutar los siguientes comandos en la carpeta del proyecto:
```
sudo docker build -t agente-ventas .
```
```
sudo docker compose up --build
```

Para terminar el programa matar el proceso de docker compose con ctrl + c y ejecutar el comando
```
sudo docker compose down -v
```
