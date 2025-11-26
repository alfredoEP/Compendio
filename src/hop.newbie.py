"""
hop.newbie.py: This script implements a Hopfield associative memory for pattern storage and retrieval.
It demonstrates pattern corruption, recovery, and performance visualization.
My first attempt at Hopfield Networks (circa 2020)
"""

__author__ = "Alfredo Espinoza"

import random
import matplotlib.pyplot as plt

# Mostrar matrices 5 x 7
def mostrar(patron):
    """
    Print a 5x7 grid representation of a pattern (list of 35 elements).
    """
    for fila in range(7):
        inicio = 5 * fila
        fin = inicio + 5
        particion = patron[inicio:fin]
        linea = ["██" if x == 1 else "  " for x in particion]
        print("".join(linea))
    print()

# Construcción de la matriz de memoria

def crear_memoria(lista_de_patrones):
    """
    Create a Hopfield memory matrix from a list of input patterns.
    """
    matriz = []
    for idx, patron in enumerate(lista_de_patrones):
        if idx == 0:
            for i, val_i in enumerate(patron):
                fila = []
                for j, val_j in enumerate(patron):
                    fila.append(0 if i == j else val_i * val_j)
                matriz.append(fila)
        else:
            for i, val_i in enumerate(patron):
                for j, val_j in enumerate(patron):
                    if i == j:
                        matriz[i][j] = 0
                    else:
                        matriz[i][j] += val_i * val_j
    return matriz

lista_de_patrones = [
 [  -1,-1, 1,-1,-1,
    -1, 1,-1, 1,-1,
     1,-1,-1,-1, 1,
     1,-1,-1,-1, 1,
     1, 1, 1, 1, 1,
     1,-1,-1,-1, 1,
     1,-1,-1,-1, 1
  ],
  [  1, 1, 1, 1, 1,
     1,-1,-1,-1,-1,
     1,-1,-1,-1,-1,
     1, 1, 1,-1,-1,
     1,-1,-1,-1,-1,
     1,-1,-1,-1,-1,
     1, 1, 1, 1, 1
   ],
  [  1, 1, 1, 1, 1,
    -1,-1, 1,-1,-1,
    -1,-1, 1,-1,-1,
    -1,-1, 1,-1,-1,
    -1,-1, 1,-1,-1,
    -1,-1, 1,-1,-1,
     1, 1, 1, 1, 1
    ],
  [ -1, 1, 1, 1,-1,
     1,-1,-1,-1, 1,
     1,-1,-1,-1, 1,
     1,-1,-1,-1, 1,
     1,-1,-1,-1, 1,
     1,-1,-1,-1, 1,
    -1, 1, 1, 1,-1
  ],
  [  1,-1,-1,-1, 1,
     1,-1,-1,-1, 1,
     1,-1,-1,-1, 1,
     1,-1,-1,-1, 1,
     1,-1,-1,-1, 1,
     1,-1,-1,-1, 1,
    -1, 1, 1, 1,-1
   ]
 ]

for i in lista_de_patrones:
  mostrar(i)

memoria = crear_memoria(lista_de_patrones)
for i in memoria: print(i)

# Recuperación

def recuperar_patron(memoria, patron, lista_de_patrones, ciclos_maximos=100):
    """
    Recover a pattern from memory given a possibly corrupted input.
    Iterates up to ciclos_maximos times or until a stored pattern is found.
    """
    x = patron[:]
    for _ in range(ciclos_maximos):
        patron_recuperado = []
        for fila in memoria:
            xij = sum(i * j for i, j in zip(fila, x))
            patron_recuperado.append(1 if xij > 0 else -1 if xij < 0 else 0)
        if patron_recuperado in lista_de_patrones:
            return patron_recuperado
        x = patron_recuperado
    return patron_recuperado

señal_corrupta = [ 1,-1, 1,-1,-1,
                  -1, 1,-1, 1,-1,
                   1,-1,-1,-1, 1,
                   1,-1, 1,-1, 1,
                   1, 1, 1, 1, 1,
                   1,-1,-1,-1, 1,
                   1,-1,-1,-1,-1]

print("Señal Corrupta")
mostrar(señal_corrupta)
señal_recuperada = recuperar_patron(memoria,señal_corrupta,lista_de_patrones,100)
mostrar(señal_recuperada)
print("Señal Recuperada")

def prueba(memoria, lista_de_patrones, número_de_veces_a_alterar, número_de_ciclos_a_iterar, imprimir, elegido):
    copia_lista_de_patrones = list.copy(lista_de_patrones)
    patrón = lista_de_patrones[elegido]
    if imprimir != "no imprimir":
        print("Patrón elegido aleatoriamente")
        mostrar(patrón)

    patrón_corrupto = list.copy(patrón)
    for _ in range(número_de_veces_a_alterar):
        índice_de_elemento_elegido_para_alterar = random.choice(list(range(len(patrón))))
        patrón_corrupto[índice_de_elemento_elegido_para_alterar] = random.choice([-1, 1])
    if imprimir != "no imprimir":
        print("Patrón elegido aleatoriamente - corrupto")
        mostrar(patrón_corrupto)

    patrón_recuperado = recuperar_patron(memoria, patrón_corrupto, copia_lista_de_patrones, número_de_ciclos_a_iterar)
    if imprimir != "no imprimir":
        print("Patrón elegido aleatoriamente - recuperado")
        mostrar(patrón_recuperado)

    cantidad_de_aciertos = sum(1 for i, j in zip(patrón, patrón_recuperado) if i == j)
    porcentaje_de_aciertos = cantidad_de_aciertos * 100 / len(patrón)

    if imprimir != "no imprimir":
        print(f"Exactitud: {porcentaje_de_aciertos}%")
    return porcentaje_de_aciertos

prueba(memoria, lista_de_patrones, 50, 100, "imprimir", 0)

for _ in range(200):
    desempeno = [prueba(memoria, lista_de_patrones, i, 100, False, 0) for i in range(0, 51, 5)]
    x = [i for i in range(0, 51, 5)]
    plt.plot(x, desempeno)
plt.xlim(0, 50)
plt.title("Patrón: A")
plt.xlabel("Nivel de Ruido")
plt.ylabel("Exactitud (%)")
plt.show()