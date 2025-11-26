# !/usr/bin/env python

__author__ = "Alfredo Espinoza"

"""
One Clause At a Time (OCAT) Algorithm
======================================

This script implements the OCAT algorithm for learning Boolean expressions
from labeled data. The algorithm reads a CSV file, binarizes the data, and
searches for clauses that can classify positive and negative examples.

Reference and further reading:
https://www.researchgate.net/publication/226973010_The_One_Clause_at_a_Time_OCAT_Approach_to_Data_Mining_and_Knowledge_Discovery
"""

import csv
import numpy as np
import random as rn
import time
import tempfile
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def parse_arff_file(arff_filename):
    """
    Parse ARFF file and extract attribute names and data.
    
    Parameters:
    -----------
    arff_filename : str
        Path to the ARFF file
    
    Returns:
    --------
    tuple
        (attribute_names, data_table) - List of attribute names and data as list of lists
    """
    attribute_names = []
    data_table = []
    in_data_section = False
    
    with open(arff_filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('%'):
                continue
            
            # Parse attribute names
            if line.lower().startswith('@attribute'):
                # Extract attribute name - handle quoted names with spaces
                # Match: @attribute 'Name with spaces' or @attribute Name
                match = re.search(r"@attribute\s+['\"]([^'\"]+)['\"]|@attribute\s+(\S+)", line, re.IGNORECASE)
                if match:
                    # Use the first matching group (quoted) or second (unquoted)
                    attr_name = match.group(1) if match.group(1) else match.group(2)
                    attribute_names.append(attr_name)
            
            # Check for data section
            elif line.lower().startswith('@data'):
                in_data_section = True
            
            # Parse data rows
            elif in_data_section:
                # Split by comma and strip whitespace
                row = [cell.strip() for cell in line.split(',')]
                data_table.append(row)
    
    return attribute_names, data_table


def load_data(filename):
    """
    Load data from CSV file into a list.
    
    Parameters:
    -----------
    filename : str
        Path to the CSV file
    
    Returns:
    --------
    list
        Data table as a list of lists
    """
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data_table = []
        for fila in csv_reader:
            concatenado = []
            for celda in fila:
                concatenado.append(celda)
            data_table.append(concatenado)
    return data_table


def preprocess_data(data_table, sort_column, has_header=True):
    """
    Preprocess the data table by sorting and extracting positive indices.
    
    Parameters:
    -----------
    data_table : list
        Raw data table
    sort_column : int
        Column index to sort by
    has_header : bool
        Whether the data table has a header row to remove
    
    Returns:
    --------
    tuple
        (dataTable, índices_positivos) - Preprocessed data and positive indices
    """
    # Remove header row if present
    if has_header:
        data_table.pop(0)
    data_table = np.array(data_table)
    
    # Sort by specified column
    sorted_data_table = data_table[np.argsort(data_table[:, sort_column])]
    
    dataTable = []
    índices_positivos = []
    
    # Extract positive indices (where last column is '1')
    for índice, i in enumerate(sorted_data_table):
        if i[-1] == '1':
            índices_positivos.append(índice)
    
    # Remove last column (labels)
    for num, i in enumerate(sorted_data_table):
        dataTable.append(i[:-1])
    
    dataTable = np.array(dataTable).astype(float)
    
    return dataTable, índices_positivos


# ============================================================================
# BINARIZATION FUNCTIONS
# ============================================================================

def obtener_valoresObservadosDiferentes(dataTable):
    """
    Extract different observed values per column from a data table.
    
    Parameters:
    -----------
    dataTable : np.ndarray
        Data table as numpy array
    
    Returns:
    --------
    list
        List of sorted unique values per column
    """
    valoresObservadosDiferentes = []
    
    for columna in list(range(len(dataTable[0]))):
        elementosColumna = []
        for fila in dataTable:
            if fila[columna] not in elementosColumna:
                elementosColumna.append(fila[columna])
        valoresObservadosDiferentes.append(elementosColumna)
    
    for (orden, arreglo) in enumerate(valoresObservadosDiferentes):
        valoresObservadosDiferentes[orden] = sorted(arreglo, reverse=False)
    
    return valoresObservadosDiferentes


def binarizacion(dataTable, valoresObservadosDiferentes):
    """
    Transform numeric data into binary vectors.
    
    Parameters:
    -----------
    dataTable : np.ndarray
        Data table with numeric values
    valoresObservadosDiferentes : list
        List of unique values per column
    
    Returns:
    --------
    list
        Binarized data table as list of numpy arrays (each attribute may have different shape)
    """
    dataTableBinarizado = []
    
    for numDeColumna, _ in enumerate(dataTable[0]):
        arregloColumnaXij = []
        for numDeFila, _ in enumerate(dataTable):
            arregloFilaXij = []
            for variableBooleana in valoresObservadosDiferentes[numDeColumna]:
                if dataTable[numDeFila][numDeColumna] >= variableBooleana:
                    arregloFilaXij.append(1)
                else:
                    arregloFilaXij.append(0)
            arregloColumnaXij.append(arregloFilaXij)
        # Keep as numpy array for each attribute
        dataTableBinarizado.append(np.array(arregloColumnaXij))
    
    # Return as list since attributes have different numbers of unique values
    return dataTableBinarizado


def division_ejemplos(dataTableBinarizado, índices_positivos):
    """
    Divide binarized data into positive and negative example sets.
    
    Parameters:
    -----------
    dataTableBinarizado : list
        Binarized data table (list of numpy arrays)
    índices_positivos : list
        List of positive example indices
    
    Returns:
    --------
    tuple
        (E_posBin, E_negBin) - Positive and negative example sets
    """
    E_posBin, E_negBin = [], []
    
    # Get number of rows from first attribute's shape
    num_rows = dataTableBinarizado[0].shape[0]
    
    for i in range(num_rows):
        if i in índices_positivos:
            # Collect row i from all attributes
            row = []
            for attr in dataTableBinarizado:
                row.append(attr[i])
            E_posBin.append(row)
        else:
            # Collect row i from all attributes
            row = []
            for attr in dataTableBinarizado:
                row.append(attr[i])
            E_negBin.append(row)
    
    E_posBin, E_negBin = np.array(E_posBin, dtype=object), np.array(E_negBin, dtype=object)
    return E_posBin, E_negBin


def transponer_columnas(E_posBinT, E_negBinT):
    """
    Transpose columns with multiple sub-columns.
    
    Parameters:
    -----------
    E_posBinT : np.ndarray
        Transposed positive examples
    E_negBinT : np.ndarray
        Transposed negative examples
    
    Returns:
    --------
    tuple
        (E_posBinTCol, E_negBinTCol) - Column-transposed arrays
    """
    columnas_pos, columnas_neg = [], []
    
    for i in list(range(E_posBinT.shape[0])):
        largo = len(E_posBinT[i][0])
        for j in list(range(largo)):
            columna = []
            for fila in E_posBinT[i]:
                columna.append(fila[j])
            columnas_pos.append(columna)
    
    for i in list(range(E_negBinT.shape[0])):
        largo = len(E_negBinT[i][0])
        for j in list(range(largo)):
            columna = []
            for fila in E_negBinT[i]:
                columna.append(fila[j])
            columnas_neg.append(columna)
    
    E_posBinTCol, E_negBinTCol = np.array(columnas_pos), np.array(columnas_neg)
    return E_posBinTCol, E_negBinTCol


# ============================================================================
# OCAT ALGORITHM FUNCTIONS
# ============================================================================

def Pos(xi, negado, E_pos):
    """
    Return list of indices with value 1 in positive examples for term xi or ¬xi.
    
    Parameters:
    -----------
    xi : int
        Column index of the attribute (starting from 0)
    negado : str
        'negado' or 'no negado' indicating if values are negated
    E_pos : np.ndarray
        Positive example set
    
    Returns:
    --------
    list
        List of indices where term is positive
    """
    arreglo = []
    
    for i, elemento in enumerate(E_pos[xi]):
        if negado == 'no negado':
            if elemento == 1:
                arreglo.append(i)
        else:
            if elemento == 0:
                arreglo.append(i)
    
    return arreglo


def Neg(xi, negado, E_neg):
    """
    Return list of indices with value 1 in negative examples for term xi or ¬xi.
    
    Parameters:
    -----------
    xi : int
        Column index of the attribute (starting from 0)
    negado : str
        'negado' or 'no negado' indicating if values are negated
    E_neg : np.ndarray
        Negative example set
    
    Returns:
    --------
    list
        List of indices where term is positive
    """
    arreglo = []
    
    for i, elemento in enumerate(E_neg[xi]):
        if negado == 'no negado':
            if elemento == 1:
                arreglo.append(i)
        else:
            if elemento == 0:
                arreglo.append(i)
    
    return arreglo


def calculate_fitness_value(ti, negado, E_pos, E_neg):
    """
    Calculate fitness value for a term considering positive and negative sets.
    
    Parameters:
    -----------
    ti : int
        Term index (starting from 0)
    negado : str
        'negado' or 'no negado'
    E_pos : np.ndarray
        Positive example set
    E_neg : np.ndarray
        Negative example set
    
    Returns:
    --------
    float
        Fitness value
    """
    cantidad_de_1s_en_E_pos = len(Pos(ti, negado, E_pos))
    cantidad_de_0s_en_E_neg = len(E_neg[0]) - len(Neg(ti, negado, E_neg))
    a = cantidad_de_1s_en_E_pos
    b = cantidad_de_0s_en_E_neg
    c = (a * b)
    return c


def build_sorted_fitness_list(E_pos, E_neg):
    """
    Create sorted list of tuples with fitness values for all terms.
    
    Parameters:
    -----------
    E_pos : np.ndarray
        Positive example set
    E_neg : np.ndarray
        Negative example set
    
    Returns:
    --------
    np.ndarray
        Array of tuples (negated, index, fitness) sorted by fitness
    """
    arreglo = []
    
    for i, fila in enumerate(E_pos):
        arreglo.append([0, i, calculate_fitness_value(i, 'no negado', E_pos, E_neg)])
        arreglo.append([1, i, calculate_fitness_value(i, 'negado', E_pos, E_neg)])
    
    tamaño_f = lambda valor: valor[2]
    arreglo.sort(key=tamaño_f, reverse=True)
    arreglo = np.array(arreglo)
    
    return arreglo


def eliminar_filas_E_pos(filas_que_Ci_vuelve_completas, E_pos_temporal):
    """
    Create new list without rows completed by clause Ci in positive examples.
    
    Parameters:
    -----------
    filas_que_Ci_vuelve_completas : list
        Indices of rows to remove
    E_pos_temporal : list
        Current positive example set
    
    Returns:
    --------
    list
        Updated positive example set
    """
    E_poss = []
    for i in E_pos_temporal:
        E_poss.append([k for j, k in enumerate(i) if j not in filas_que_Ci_vuelve_completas])
    return E_poss


def eliminar_filas_E_neg(filas_que_Ci_vuelve_consistentes, E_neg_temporal):
    """
    Create new list without rows made consistent by clause Ci in negative examples.
    
    Parameters:
    -----------
    filas_que_Ci_vuelve_consistentes : set
        Indices of rows to remove
    E_neg_temporal : list
        Current negative example set
    
    Returns:
    --------
    list
        Updated negative example set
    """
    E_negg = []
    for i in E_neg_temporal:
        E_negg.append([k for j, k in enumerate(i) if j not in filas_que_Ci_vuelve_consistentes])
    return E_negg


def OCAT(dataTable, índices_positivos, fracción_de_m_elegidos, tiempo_máximo_de_cómputo):
    """
    One Clause At a Time algorithm for learning Boolean expressions.
    
    Parameters:
    -----------
    dataTable : np.ndarray
        Data table with numeric values
    índices_positivos : list
        List of positive example indices
    fracción_de_m_elegidos : float
        Fraction of top candidates to consider (0-1)
    tiempo_máximo_de_cómputo : float
        Maximum computation time in seconds
    
    Returns:
    --------
    tuple
        (C, valoresObservadosDiferentes) - Learned clauses and observed values
    """
    # Binarization
    valoresObservadosDiferentes = obtener_valoresObservadosDiferentes(dataTable)
    dataTableBinarizado = binarizacion(dataTable, valoresObservadosDiferentes)
    E_posBin, E_negBin = division_ejemplos(dataTableBinarizado, índices_positivos)
    E_posBinT, E_negBinT = np.matrix.transpose(E_posBin), np.matrix.transpose(E_negBin)
    E_posBinTCol, E_negBinTCol = transponer_columnas(E_posBinT, E_negBinT)
    E_pos_temporal, E_neg_temporal = np.ndarray.tolist(np.copy(E_posBinTCol)), np.ndarray.tolist(np.copy(E_negBinTCol))
    
    start_time = time.time()
    
    # OCAT clause search
    C = []
    filas_que_C_vuelve_consistentes = set()
    
    while len(E_neg_temporal[0]) > 0:
        E_pos_temporal = E_posBinTCol
        Ci = []
        
        while len(E_pos_temporal[0]) > 0:
            L = build_sorted_fitness_list(E_pos_temporal, E_neg_temporal)
            L = np.array(L)
            
            # Reduce list to top 'm' candidates
            m = int(len(L) * fracción_de_m_elegidos)
            L_reducida = L[:m, :]
            
            # Select a term randomly from reduced list
            término_seleccionado = rn.choice(L_reducida)
            
            if término_seleccionado[0] == 0:
                t = "x{}".format((término_seleccionado[1] + 1).astype(int))
            else:
                t = "¬x{}".format((término_seleccionado[1] + 1).astype(int))
            
            if not Ci:
                Ci.append(t)
            else:
                Ci.append('V')
                Ci.append(t)
            
            # Update E+ by removing completed rows
            if término_seleccionado[0] == 0:
                filas_que_Ci_vuelve_completas = Pos(int(término_seleccionado[1]), "no negado", E_pos_temporal)
            else:
                filas_que_Ci_vuelve_completas = Pos(int(término_seleccionado[1]), "negado", E_pos_temporal)
            
            E_pos_temporal = eliminar_filas_E_pos(filas_que_Ci_vuelve_completas, E_pos_temporal)
        
        # Update E- by checking consistency
        filas_que_Ci_vuelve_consistentes = set()
        for i, t in enumerate(Ci):
            if t != 'V':
                if '¬' in t:
                    filas_que_t_vuelve_consistentes = set(Neg((int(t.strip("¬x")) - 1), "no negado", E_neg_temporal))
                else:
                    filas_que_t_vuelve_consistentes = set(Neg((int(t.strip("¬x")) - 1), "negado", E_neg_temporal))
                
                if i == 0:
                    filas_que_Ci_vuelve_consistentes = filas_que_t_vuelve_consistentes
                else:
                    filas_que_Ci_vuelve_consistentes = filas_que_Ci_vuelve_consistentes.intersection(filas_que_t_vuelve_consistentes)
        
        if filas_que_C_vuelve_consistentes.union(filas_que_Ci_vuelve_consistentes) > filas_que_C_vuelve_consistentes:
            if not C:
                print("#### Agregando cláusula Ci a C.   {}".format(Ci))
                C.append(Ci)
            else:
                C.append('Λ')
                C.append(Ci)
            filas_que_C_vuelve_consistentes = filas_que_C_vuelve_consistentes.union(filas_que_Ci_vuelve_consistentes)
            E_neg_temporal = eliminar_filas_E_neg(filas_que_C_vuelve_consistentes, E_neg_temporal)
            
            # Print C only when it changes (clause was added)
            print("### Cantidad de Filas y Filas que C vuelve consistentes : {}\n{}\n".format(
                len(E_negBinT[0]) - len(E_neg_temporal[0]), sorted(list(filas_que_C_vuelve_consistentes))))
            print("███ C : \n{}".format([i for i in C]))
        
        ratito = time.time() - start_time
        if ratito > tiempo_máximo_de_cómputo:
            print("Tiempo máximo de búsqueda excedido!!! Vuelva a ejecutar este bloque de código.")
            break
    
    if ratito <= tiempo_máximo_de_cómputo:
        print("# E_neg_temporal está vacío.")
        print("\n¡ÉXITO!\n")
        print("\n  C :")
    
    return C, valoresObservadosDiferentes


def evaluate_instance(instance, C, valoresObservadosDiferentes):
    """
    Evaluate a single instance using the learned OCAT model.
    
    Parameters:
    -----------
    instance : list or tuple
        A data instance with numeric values (excluding the label)
    C : list
        Learned clauses from OCAT algorithm
    valoresObservadosDiferentes : list
        Observed unique values per column
    
    Returns:
    --------
    int
        1 if the instance satisfies the model (positive prediction), 0 otherwise
    """
    # Get clean clauses (without 'Λ')
    Copia_C = [clause for clause in C if clause != 'Λ']
    
    # The model is a conjunction (AND) of clauses
    # All clauses must be satisfied for the instance to be positive
    for clause in Copia_C:
        # Each clause is a disjunction (OR) of terms
        # At least one term must be satisfied for the clause to be true
        clause_satisfied = False
        
        for term in clause:
            if term == 'V':  # Skip disjunction operators
                continue
            
            # Parse the term
            is_negated = '¬' in term
            term_num = int(term.strip("¬x")) - 1
            
            # Map term_num to (attribute_index, threshold_index)
            current_term = 0
            found = False
            
            for attr_idx in range(len(valoresObservadosDiferentes)):
                for threshold_idx, threshold in enumerate(valoresObservadosDiferentes[attr_idx]):
                    if current_term == term_num:
                        # Evaluate the term
                        attribute_value = instance[attr_idx]
                        
                        # xN means: attribute >= threshold
                        if is_negated:
                            # ¬xN means: attribute < threshold
                            term_value = attribute_value < threshold
                        else:
                            # xN means: attribute >= threshold
                            term_value = attribute_value >= threshold
                        
                        if term_value:
                            clause_satisfied = True
                            break
                        found = True
                        break
                    current_term += 1
                if found:
                    break
            
            if clause_satisfied:
                break
        
        # If any clause is not satisfied, return 0 (negative)
        if not clause_satisfied:
            return 0
    
    # All clauses satisfied, return 1 (positive)
    return 1


def Reglas_C(C, valoresObservadosDiferentes, attribute_names=None):
    """
    Convert learned clauses into human-readable rules.
    
    Parameters:
    -----------
    C : list
        Learned clauses
    valoresObservadosDiferentes : list
        Observed unique values per column
    attribute_names : list, optional
        List of attribute names for readable output
    
    Returns:
    --------
    list
        List of rules in readable format
    """
    # Copy C
    Copia_C = C.copy()
    Copia_C = [i for i in Copia_C if i != 'Λ']
    
    # Remove 'Λ' and 'V' symbols
    CLimpio = []
    for i in Copia_C:
        CLimpio.append([j for j in i if j != 'V'])
    
    # Remove 'x' and '¬' to get numeric indices
    CNumérico = CLimpio.copy()
    for i, lista in enumerate(CNumérico):
        for j, elemento in enumerate(lista):
            if '¬' not in elemento:
                CNumérico[i][j] = int(elemento.strip("x")) - 1
            else:
                CNumérico[i][j] = '¬' + str(int(elemento.strip("¬x")) - 1)
    
    Reglas = []
    for cláusula in CNumérico:
        regla = []
        for elemento in cláusula:
            x = str(elemento)
            
            if '¬' not in x:
                x = int(x)
                negado = False
            else:
                x = int(x.strip("¬"))
                negado = True
            
            for ni, i in enumerate(valoresObservadosDiferentes):
                for j in i:
                    if x == 0:
                        y = j
                        break
                    else:
                        x -= 1
                else:
                    continue
                break
            
            elemento = str(elemento)
            elemento = elemento.strip("¬")
            
            # Use attribute name if available, otherwise use x notation
            if attribute_names and ni < len(attribute_names):
                attr_label = attribute_names[ni]
            else:
                attr_label = f"x{ni}"
            
            if negado == False:
                regla.append("{}>={}".format(attr_label, y))
            else:
                regla.append("{}<{}".format(attr_label, y))
        Reglas.append(regla)
    
    Reglas = [sorted(i) for i in Reglas]
    return Reglas


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("ONE CLAUSE AT A TIME (OCAT) ALGORITHM")
    print("=" * 70)
    
    # Configuration
    # https://archive.ics.uci.edu/static/public/472/caesarian+section+classification+dataset.zip
    ARFF_FILE = './data/caesarian.csv.arff'
    MAX_COMPUTATION_TIME = 15  # seconds
    M_FRACTION = 0.4  # Fraction of top candidates to consider
    
    # Load and parse ARFF file
    print("\n[Step 1] Loading Data from ARFF File")
    print("-" * 70)
    attribute_names, data_table = parse_arff_file(ARFF_FILE)
    
    # Display attribute information
    print(f"Attributes found: {', '.join(attribute_names)}")
    print(f"Total attributes: {len(attribute_names)}")
    print(f"Total data rows: {len(data_table)}")
    
    # Display value ranges per attribute
    print("\nValue ranges per attribute:")
    for i, attr_name in enumerate(attribute_names[:-1]):
        values = [float(row[i]) for row in data_table]
        min_val = min(values)
        max_val = max(values)
        print(f"  {i}: {attr_name:25s} [{min_val:.0f} : {max_val:.0f}]")
    
    # Create temporary CSV file for processing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as tmp_file:
        csv_writer = csv.writer(tmp_file)
        # Write attribute names as header
        csv_writer.writerow(attribute_names)
        # Write data rows
        csv_writer.writerows(data_table)
        tmp_csv_path = tmp_file.name
    
    try:
        # Load from temporary CSV
        data_table_with_header = load_data(tmp_csv_path)
        
        # Get user input for sorting column
        print("\nAvailable attributes for sorting:")
        for i, attr in enumerate(attribute_names[:-1]):  # Exclude last column (label)
            print(f"  {i}: {attr}")
        
        columnas_a_elegir = len(attribute_names) - 1
        elección = input(f"\nElija un número entre 0 y {columnas_a_elegir - 1}  :  ")
        
        dataTable, índices_positivos = preprocess_data(data_table_with_header, int(elección), has_header=True)
        print(f"Data shape: {dataTable.shape}")
        print(f"Positive examples: {len(índices_positivos)}")
        print(f"Negative examples: {len(dataTable) - len(índices_positivos)}")
        
        # Run OCAT algorithm
        print("\n[Step 2] Running OCAT Algorithm")
        print("-" * 70)
        C, valoresObservadosDiferentes = OCAT(
            dataTable,
            índices_positivos,
            fracción_de_m_elegidos=M_FRACTION,
            tiempo_máximo_de_cómputo=MAX_COMPUTATION_TIME
        )
        
        # Build a complete term-to-rule mapping for readable output
        term_to_rule_map = {}
        term_index = 0
        
        for attr_idx, attr_name in enumerate(attribute_names[:-1]):
            unique_values = valoresObservadosDiferentes[attr_idx]
            for threshold in unique_values:
                # Round threshold to int for readability
                threshold_int = int(round(threshold))
                term_to_rule_map[term_index] = (attr_name, threshold_int)
                term_index += 1
        
        # Display learned clauses in readable format
        print("\n[Step 3] Learned Clauses (Readable Format)")
        print("-" * 70)
        
        Copia_C_display = [clause for clause in C if clause != 'Λ']
        for clause_idx, clause in enumerate(Copia_C_display):
            if clause_idx > 0:
                print("\nAND\n")
            print(f"Clause {clause_idx + 1}: [", end="")
            readable_terms = []
            for term in clause:
                if term != 'V':
                    is_negated = '¬' in term
                    term_num = int(term.strip("¬x")) - 1
                    
                    if term_num in term_to_rule_map:
                        attr_name, threshold_int = term_to_rule_map[term_num]
                        if is_negated:
                            readable_terms.append(f"{attr_name}<{threshold_int}")
                        else:
                            readable_terms.append(f"{attr_name}>={threshold_int}")
            print(" OR ".join(readable_terms) + " ]")
        
        # Display detailed clause breakdown
        print("\n[Step 4] Detailed Clause Breakdown")
        print("-" * 70)
        
        # Get clean clauses (without 'Λ')
        Copia_C = [clause for clause in C if clause != 'Λ']
        
        for clause_idx, clause in enumerate(Copia_C):
            print(f"\nClause {clause_idx + 1} terms:")
            # Process each term in the clause
            for term in clause:
                if term != 'V':
                    is_negated = '¬' in term
                    term_num = int(term.strip("¬x")) - 1
                    
                    if term_num in term_to_rule_map:
                        attr_name, threshold_int = term_to_rule_map[term_num]
                        
                        # If negated, flip the operator
                        if is_negated:
                            displayed_rule = f"{attr_name}<{threshold_int}"
                        else:
                            displayed_rule = f"{attr_name}>={threshold_int}"
                        
                        print(f"  • {displayed_rule}")
                    else:
                        print(f"  • {term} : (term index out of range)")
        
        # Test the model on the dataset
        print("\n[Step 5] Model Testing")
        print("-" * 70)
        
        # Evaluate all instances
        correct_predictions = 0
        total_instances = len(dataTable)
        
        predictions = []
        for i, instance in enumerate(dataTable):
            prediction = evaluate_instance(instance, C, valoresObservadosDiferentes)
            predictions.append(prediction)
            
            # Compare with actual label (1 if index in positive indices, 0 otherwise)
            actual = 1 if i in índices_positivos else 0
            if prediction == actual:
                correct_predictions += 1
        
        accuracy = (correct_predictions / total_instances) * 100
        
        print(f"Total instances: {total_instances}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Incorrect predictions: {total_instances - correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Confusion matrix
        true_positives = sum(1 for i in range(len(predictions)) if predictions[i] == 1 and i in índices_positivos)
        false_positives = sum(1 for i in range(len(predictions)) if predictions[i] == 1 and i not in índices_positivos)
        true_negatives = sum(1 for i in range(len(predictions)) if predictions[i] == 0 and i not in índices_positivos)
        false_negatives = sum(1 for i in range(len(predictions)) if predictions[i] == 0 and i in índices_positivos)
        
        print("\nConfusion Matrix:")
        print(f"  True Positives:  {true_positives}")
        print(f"  False Positives: {false_positives}")
        print(f"  True Negatives:  {true_negatives}")
        print(f"  False Negatives: {false_negatives}")
        
        if true_positives + false_positives > 0:
            precision = (true_positives / (true_positives + false_positives)) * 100
            print(f"\nPrecision: {precision:.2f}%")
        
        if true_positives + false_negatives > 0:
            recall = (true_positives / (true_positives + false_negatives)) * 100
            print(f"Recall: {recall:.2f}%")
        
        # Generate confusion matrix visualization
        print("\n[Step 6] Generating Confusion Matrix Visualization")
        print("-" * 70)
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(14, 10))
        
        # Confusion Matrix subplot
        ax1 = plt.subplot(2, 1, 1)
        
        # Create confusion matrix data
        conf_matrix = np.array([[true_positives, false_negatives],
                                [false_positives, true_negatives]])
        
        # Display the confusion matrix
        im = ax1.imshow(conf_matrix, cmap='Blues', alpha=0.6)
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax1.text(j, i, conf_matrix[i, j],
                               ha="center", va="center", color="black", fontsize=20, fontweight='bold')
        
        # Set labels and title
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Predicted Positive', 'Predicted Negative'], fontsize=11)
        ax1.set_yticklabels(['Actual Positive', 'Actual Negative'], fontsize=11)
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        
        # Add grid
        ax1.set_xticks([0.5], minor=True)
        ax1.set_yticks([0.5], minor=True)
        ax1.grid(which='minor', color='black', linestyle='-', linewidth=2)
        
        # Add metrics text
        metrics_text = f"Accuracy: {accuracy:.2f}%\n"
        if true_positives + false_positives > 0:
            metrics_text += f"Precision: {precision:.2f}%\n"
        if true_positives + false_negatives > 0:
            metrics_text += f"Recall: {recall:.2f}%"
        
        ax1.text(1.15, 0.5, metrics_text, transform=ax1.transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Learned Clauses subplot
        ax2 = plt.subplot(2, 1, 2)
        ax2.axis('off')
        
        # Format clauses for display in readable format
        clause_text = "Learned Clauses (Boolean Expression):\n\n"
        
        Copia_C_viz = [clause for clause in C if clause != 'Λ']
        for clause_idx, clause in enumerate(Copia_C_viz):
            if clause_idx > 0:
                clause_text += "\n  AND\n\n"
            
            clause_text += f"Clause {clause_idx + 1}: "
            readable_terms = []
            
            for term in clause:
                if term != 'V':
                    is_negated = '¬' in term
                    term_num = int(term.strip("¬x")) - 1
                    
                    if term_num in term_to_rule_map:
                        attr_name, threshold_int = term_to_rule_map[term_num]
                        if is_negated:
                            readable_terms.append(f"{attr_name}<{threshold_int}")
                        else:
                            readable_terms.append(f"{attr_name}>={threshold_int}")
            
            clause_text += "  OR  ".join(readable_terms) + "\n"
        
        # Display clause text
        ax2.text(0.05, 0.95, clause_text, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('./assets/OCAT_results.png', dpi=150, bbox_inches='tight')
        print("✓ Confusion matrix saved as './assets/OCAT_results.png'")
        plt.close()
        
        print("\n" + "=" * 70)
        print("OCAT ALGORITHM COMPLETED")
        print("=" * 70)
    
    finally:
        # Clean up temporary CSV file
        import os
        if os.path.exists(tmp_csv_path):
            os.unlink(tmp_csv_path)
            print(f"\nTemporary file cleaned up.")
