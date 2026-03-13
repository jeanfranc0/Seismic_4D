import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap, BoundaryNorm


def reshape_to_volumes(df_ip, df_amp):
    """
    Transforma el DataFrame plano en tensores 2D por cada Inline.
    Resultado esperado: (N_inlines, N_samples_time, N_xlines)
    """  
    assert df_ip[['inline', 'xline', 'time']].equals(df_amp[['inline', 'xline', 'time']])
    
    # Identificamos las dimensiones únicas
    inlines = sorted(df_ip['inline'].unique())
    xlines = sorted(df_ip['xline'].unique())
    times = sorted(df_ip['time'].unique())
    
    n_inlines = len(inlines)
    n_xlines = len(xlines)
    n_times = len(times)
    
    print(f"Dimensiones detectadas: {n_inlines} Inlines, {n_xlines} Xlines, {n_times} Samples de tiempo.")

    # Creamos contenedores vacíos para los volúmenes
    # Forma: (N_Inlines, Time, Xline) -> Similar a una imagen (N, Alto, Ancho)
    X_vol = np.zeros((n_inlines, n_times, n_xlines))
    y_vol = np.zeros((n_inlines, n_times, n_xlines))

    # Unimos los datos para asegurar correspondencia
    df_combined = pd.merge(df_ip, df_amp, on=['inline', 'xline', 'time'])

    # Llenamos los volúmenes
    for i, idx_inline in enumerate(inlines):
        # Extraemos la sección 2D
        section = df_combined[df_combined['inline'] == idx_inline]
        
        # Pivotamos para organizar Tiempo en filas y Xline en columnas
        X_slice = section.pivot(index='time', columns='xline', values=df_ip.columns[-1]).values
        y_slice = section.pivot(index='time', columns='xline', values=df_amp.columns[-1]).values
        
        X_vol[i, :, :] = X_slice
        y_vol[i, :, :] = y_slice
    
    return X_vol, y_vol


def preprocessing_shape(train_val_ip, train_val_amp, column_name_01, test_ip, test_amp, column_name_77):
    # 1. Carga (Asumiendo que ya tienes los CSVs del paso anterior)
    # Entrenamos con Model 001 [cite: 6, 25]
    X_train_full, y_train_full = reshape_to_volumes(train_val_ip, train_val_amp, column_name_01)

    # Testeamos con Model 077 [cite: 7, 26]
    X_test, y_test = reshape_to_volumes(test_ip, test_amp, column_name_77)

    # 2. Split de entrenamiento y validación conservando SECCIONES COMPLETAS
    # Esto evita que el modelo vea partes de una imagen en entrenamiento y otras en validación
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    print(f"\nForma final de X_train (N, H, W): {X_train.shape}")
    print(f"\nForma final de X_val (N, H, W): {X_val.shape}")
    print(f"\nForma final de X_test (N, H, W): {X_test.shape}")
    print(f"\nForma final de y_train (N, H, W): {y_train.shape}")
    print(f"\nForma final de y_val (N, H, W): {y_val.shape}")
    print(f"\nForma final de y_test (N, H, W): {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_data():
    # --- CARGA DE DATOS ---
    # Nota: Cargamos Model01 para Train/Val y Model77 para Test
    train_val_ip = pd.read_csv("/home/users/jeanfranco.escobedo/jeanfranco_2025/Jeanfranco/data/raw_dfs/Model01_dIP.csv")
    train_val_amp = pd.read_csv("/home/users/jeanfranco.escobedo/jeanfranco_2025/Jeanfranco/data/raw_dfs/Model01_dAMP.csv")

    test_ip = pd.read_csv("/home/users/jeanfranco.escobedo/jeanfranco_2025/Jeanfranco/data/raw_dfs/Model77_dIP.csv")
    test_amp = pd.read_csv("/home/users/jeanfranco.escobedo/jeanfranco_2025/Jeanfranco/data/raw_dfs/Model77_dAMP.csv")

    return train_val_ip, train_val_amp, test_ip, test_amp


def plot_seismic_slice_reshaped(X_data, value_col, sample_idx, title):
    """
    Visualizes a specific sample (N) from reshaped seismic data (N, H, W, C).
    Values are visualized with Red (negative), White (zero), and Blue (positive).
    
    Parameters:
    - X_data: 4D numpy array with shape (N, H, W, C)
    - value_col: Column name for value to display (but not used here since input is reshaped data)
    - sample_idx: Index of the sample to visualize (N value)
    - title: Title for the plot
    """
    # Get the specific sample (N) from the data
    sample_data = X_data[sample_idx, :, :]
    
    # Create the plot for the selected sample
    plt.figure(figsize=(12, 6))
    
    # Normalize the data for visualization
    norm = plt.Normalize(vmin=-np.std(sample_data)*3, vmax=np.std(sample_data)*3)
    label_cmap = ListedColormap(["red", "white", "blue"])

    # Display the image (seismic slice)
    plt.imshow(sample_data, aspect='auto', cmap=label_cmap, norm=norm,
               extent=[0, sample_data.shape[1], sample_data.shape[0], 0])

    plt.colorbar(label=value_col)
    plt.title(f"{title} - Sample: {sample_idx}")
    plt.xlabel("Crossline (W)")
    plt.ylabel("Time (H)")
    plt.savefig(title)
    plt.show()


def main():
    # --- CARGA DE DATOS ---
    train_val_ip, train_val_amp, test_ip, test_amp = load_data()

    # --- PROCESAMIENTO ---
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessing_shape(train_val_ip, train_val_amp, "Model01", test_ip, test_amp, "Model77")
    
    # --- VISUALIZACIÓN DE UNA SECCIÓN SÍSMICA ---
    sample_idx = 10  # Choose the sample index you want to visualize
    plot_seismic_slice_reshaped(X_train, 'dIP', sample_idx, "Actual Seismic Slice X_train" + str(sample_idx))
    plot_seismic_slice_reshaped(y_train, 'dIP', sample_idx, "Actual Seismic Slice y_train" + str(sample_idx))

if __name__ == "__main__":
    main()