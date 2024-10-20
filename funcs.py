import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

# Configuración de tema para las gráficas
graph_theme = "seaborn-v0_8-deep"
color_map = "Set1"

plt.style.use(graph_theme)


def moving_average(x: np.ndarray, window_size: int) -> np.ndarray:
    return np.array(
        [
            np.mean(x[i : i + window_size])
            for i in range(len(x) - window_size + 1)
        ]
    )


def plot_by_directory(
    audio_dir: str,
    csv_path: str,
    title: str,
    num_audios_to_plot: int,
    segment: tuple[float, float],
    window_size: int,
    show_errors: bool = False,
    show_combined_moving_average: bool = True,
):
    """
    Genera una gráfica de frecuencia vs amplitud para los archivos de audio en cada subdirectorio de un directorio dado.

    Args:
        audio_dir (str): Directorio que contiene los subdirectorios con archivos de audio.
        csv_path (str): Ruta al archivo CSV que contiene los nombres de los archivos de audio.
        title (str): Título de la gráfica.
        num_audios_to_plot (int): Número máximo de archivos de audio a graficar.
        segment (tuple[float, float]): Segmento de tiempo (en segundos) del audio a analizar.
        window_size (int): Tamaño de la ventana para el promedio móvil.
        show_errors (bool, optional): Si es True, muestra errores de archivos faltantes o audios demasiado cortos. Por defecto es False.
        show_combined_moving_average (bool, optional): Si es True, muestra la línea de media móvil combinada para todos los subdirectorios. Por defecto es True.
    """
    csv_data = pd.read_csv(csv_path)
    total_audios_available = len(csv_data)
    if num_audios_to_plot > total_audios_available:
        num_audios_to_plot = total_audios_available

    missing_files = {}
    short_audios = {}

    subdirectories = [
        d
        for d in os.listdir(audio_dir)
        if os.path.isdir(os.path.join(audio_dir, d))
    ]
    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap(color_map)(np.linspace(0, 1, len(subdirectories)))

    all_avg_magnitudes = []
    all_avg_frequencies = []
    total_plotted_audios_per_category = {}

    for idx, subdirectory in enumerate(subdirectories):
        category_path = os.path.join(audio_dir, subdirectory)
        all_frequencies = []
        all_magnitudes = []

        subset = csv_data.head(num_audios_to_plot)
        audio_files = subset["nombre"].values

        total_plotted_audios_per_category[subdirectory] = 0

        for audio_file in audio_files:
            audio_path = os.path.join(category_path, audio_file)

            if not os.path.exists(audio_path):
                if subdirectory not in missing_files:
                    missing_files[subdirectory] = []
                missing_files[subdirectory].append(audio_file)
                continue

            y, sr = librosa.load(audio_path)
            audio_duration = librosa.get_duration(y=y, sr=sr)
            if audio_duration < segment[1]:
                if subdirectory not in short_audios:
                    short_audios[subdirectory] = []
                short_audios[subdirectory].append(audio_file)
                continue

            total_plotted_audios_per_category[subdirectory] += 1

            start_sample = int(segment[0] * sr)
            end_sample = int(segment[1] * sr)
            y = y[start_sample:end_sample]

            fft_result = np.fft.fft(y)
            frequencies = np.fft.fftfreq(len(fft_result), 1 / sr)
            magnitude = np.abs(fft_result)

            positive_frequencies = frequencies[: len(frequencies) // 2]
            positive_magnitude_db = 20 * np.log10(
                magnitude[: len(magnitude) // 2] + 1e-9
            )

            all_frequencies.append(positive_frequencies)
            all_magnitudes.append(positive_magnitude_db)

            plt.plot(
                positive_frequencies,
                positive_magnitude_db,
                color=colors[idx],
                linewidth=0.1,
                alpha=0.02,
                zorder=2,
            )

        if all_frequencies:
            min_length = min(len(f) for f in all_frequencies)
            all_frequencies = [f[:min_length] for f in all_frequencies]
            all_magnitudes = [m[:min_length] for m in all_magnitudes]

            avg_frequencies = np.mean(all_frequencies, axis=0)
            avg_magnitude_db = np.mean(all_magnitudes, axis=0)

            all_avg_frequencies.append(avg_frequencies)
            all_avg_magnitudes.append(avg_magnitude_db)

            smoothed_magnitude_db = moving_average(
                avg_magnitude_db, window_size
            )
            plt.plot(
                avg_frequencies[: len(smoothed_magnitude_db)],
                smoothed_magnitude_db,
                label=f"{subdirectory}",
                color=colors[idx],
                linewidth=1.5,
                zorder=4,
            )

    if all_avg_magnitudes and show_combined_moving_average:
        combined_avg_magnitudes = np.mean(all_avg_magnitudes, axis=0)
        combined_avg_frequencies = np.mean(all_avg_frequencies, axis=0)
        combined_smoothed_magnitude_db = moving_average(
            combined_avg_magnitudes, window_size
        )
        plt.plot(
            combined_avg_frequencies[: len(combined_smoothed_magnitude_db)],
            combined_smoothed_magnitude_db,
            label="Media movil total",
            color="black",
            linewidth=1.5,
            linestyle="--",
            zorder=5,
        )

    plt.title(title)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud (dB)")
    plt.grid(True, zorder=1)
    if len(all_avg_frequencies) > 0 and isinstance(
        all_avg_frequencies[0], np.ndarray
    ):
        plt.xlim(20, None)
        plt.xticks(
            [20]
            + list(
                np.arange(
                    500,
                    max([np.max(f) for f in all_avg_frequencies]),
                    step=500,
                )
            ),
            rotation=45,
        )
    plt.legend(loc="upper right", fontsize="small")

    total_plotted_audios = sum(total_plotted_audios_per_category.values())
    plt.annotate(
        f"Directorio: {audio_dir}\nAudios graficados: {total_plotted_audios}"
        f"\nSegmento: {segment[1] - segment[0]} s ({segment[0]}-{segment[1]} s)",
        xy=(0.02, 0.05),
        xycoords="axes fraction",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"
        ),
    )

    annotation_text = "\n".join(
        [
            f"{subdirectory}: {count} audios"
            for subdirectory, count in total_plotted_audios_per_category.items()
        ]
    )

    plt.annotate(
        f"Audios por subdirectorio:\n{annotation_text}",
        xy=(0.25, 0.05),
        xycoords="axes fraction",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="blue", facecolor="lightyellow"
        ),
    )

    plt.tight_layout()
    plt.show()

    if show_errors:
        if missing_files:
            print(
                f"Cantidad de audios ausentes: "
                f"{sum(len(v) for v in missing_files.values())}"
            )
            for directory, files in missing_files.items():
                for archivo in files:
                    print(f"Archivo no encontrado en '{directory}': {archivo}")

        if short_audios:
            print(
                f"Audios demasiado cortos: "
                f"{sum(len(v) for v in short_audios.values())}"
            )
            for directory, files in short_audios.items():
                for audio in files:
                    print(f"Longitud insuficiente en '{directory}': {audio}")


def plot_by_category(
    audio_dir: str,
    csv_path: str,
    category_column: str,
    title: str,
    num_audios_to_plot: int,
    segment: tuple[float, float],
    window_size: int,
    show_errors: bool = False,
):
    """
    Genera una gráfica de frecuencia vs amplitud para los archivos de audio agrupados por categoría en el csv suminsitrado.

    Args:
        audio_dir (str): Directorio que contiene los archivos de audio.
        csv_path (str): Ruta al archivo CSV que contiene los nombres de los archivos de audio y sus categorías.
        category_column (str): Nombre de la columna en el CSV que contiene la categoría de cada archivo de audio.
        title (str): Título de la gráfica.
        num_audios_to_plot (int): Número máximo de archivos de audio a graficar.
        segment (tuple[float, float]): Segmento de tiempo (en segundos) del audio a analizar.
        window_size (int): Tamaño de la ventana para el promedio móvil.
        show_errors (bool, optional): Si es True, muestra errores de archivos faltantes o audios demasiado cortos. Por defecto es False.
    """
    csv_data = pd.read_csv(csv_path)
    total_audios_available = len(csv_data)
    if num_audios_to_plot > total_audios_available:
        num_audios_to_plot = total_audios_available

    missing_files = set()
    short_audios = []

    categories = csv_data[category_column].unique()
    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap(color_map)(np.linspace(0, 1, len(categories)))

    all_avg_frequencies = []
    all_avg_magnitudes = []
    total_plotted_audios = 0

    for idx, category in enumerate(categories):
        subset = csv_data[csv_data[category_column] == category]
        all_frequencies = []
        all_magnitudes = []

        for _, row in subset.iterrows():
            if total_plotted_audios >= num_audios_to_plot:
                break

            audio_file = row["nombre"]
            audio_path = os.path.join(audio_dir, audio_file)

            if not os.path.exists(audio_path):
                missing_files.add(audio_file)
                continue

            y, sr = librosa.load(audio_path)
            audio_duration = librosa.get_duration(y=y, sr=sr)
            if audio_duration < segment[1]:
                short_audios.append(audio_file)
                continue

            total_plotted_audios += 1

            start_sample = int(segment[0] * sr)
            end_sample = int(segment[1] * sr)
            y = y[start_sample:end_sample]

            fft_result = np.fft.fft(y)
            frequencies = np.fft.fftfreq(len(fft_result), 1 / sr)
            magnitude = np.abs(fft_result)

            positive_frequencies = frequencies[: len(frequencies) // 2]
            positive_magnitude_db = 20 * np.log10(
                magnitude[: len(magnitude) // 2] + 1e-9
            )

            all_frequencies.append(positive_frequencies)
            all_magnitudes.append(positive_magnitude_db)

            plt.plot(
                positive_frequencies,
                positive_magnitude_db,
                color=colors[idx],
                linewidth=0.1,
                alpha=0.01,
            )

        if all_frequencies:
            min_length = min(len(f) for f in all_frequencies)
            all_frequencies = [f[:min_length] for f in all_frequencies]
            all_magnitudes = [m[:min_length] for m in all_magnitudes]

            avg_frequencies = np.mean(all_frequencies, axis=0)
            avg_magnitude_db = np.mean(all_magnitudes, axis=0)

            all_avg_frequencies.append(avg_frequencies)
            all_avg_magnitudes.append(avg_magnitude_db)

    for idx, (avg_frequencies, avg_magnitude_db) in enumerate(
        zip(all_avg_frequencies, all_avg_magnitudes)
    ):
        smoothed_magnitude_db = moving_average(avg_magnitude_db, window_size)
        plt.plot(
            avg_frequencies[: len(smoothed_magnitude_db)],
            smoothed_magnitude_db,
            label=f"{categories[idx]}",
            color=colors[idx],
            linewidth=1.5,
        )

    plt.title(title)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud (dB)")
    plt.grid(True)
    plt.xlim(20, None)
    plt.xticks(
        [20] + list(np.arange(500, max(avg_frequencies), step=500)),
        rotation=45,
    )
    plt.legend(loc="upper right", fontsize="small")

    plt.annotate(
        f"Directorio: {audio_dir}\nAudios graficados: {total_plotted_audios}"
        f"\nSegmento: {segment[1] - segment[0]} s ({segment[0]}-{segment[1]} s)",
        xy=(0.02, 0.05),
        xycoords="axes fraction",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"
        ),
    )

    plt.tight_layout()
    plt.show()

    if show_errors:
        if missing_files:
            print(f"Audios ausentes: {len(missing_files)}")
            for archivo in missing_files:
                print(f"Archivo no encontrado: {archivo}")

        if short_audios:
            print(f"Audios demasiado cortos: {len(short_audios)}")
            for audio in short_audios:
                print(f"Longitud insuficiente: {audio}")


def plot_by_category_combinations(
    audio_dir: str,
    csv_path: str,
    category_columns: list[str],
    title: str,
    num_audios_to_plot: int,
    segment: tuple[float, float],
    window_size: int,
    show_errors: bool = False,
):
    """
    Genera una gráfica de frecuencia vs amplitud para combinaciones de categorías especificadas.

    Args:
        audio_dir (str): Directorio que contiene los archivos de audio.
        csv_path (str): Ruta al archivo CSV que contiene los nombres de los archivos de audio y sus categorías.
        category_columns (list[str]): Lista de nombres de columnas en el CSV que contienen las categorías a combinar.
        title (str): Título de la gráfica.
        num_audios_to_plot (int): Número máximo de archivos de audio a graficar.
        segment (tuple[float, float]): Segmento de tiempo (en segundos) del audio a analizar.
        window_size (int): Tamaño de la ventana para el promedio móvil.
        show_errors (bool, optional): Si es True, muestra errores de archivos faltantes o audios demasiado cortos. Por defecto es False.
    """
    csv_data = pd.read_csv(csv_path)
    total_audios_available = len(csv_data)
    if num_audios_to_plot > total_audios_available:
        num_audios_to_plot = total_audios_available

    missing_files = set()
    short_audios = []

    # Crear una nueva columna con la combinación de las categorías especificadas
    csv_data["combined_category"] = csv_data[category_columns].apply(
        lambda x: "-".join(x.astype(str)), axis=1
    )
    combined_categories = csv_data["combined_category"].unique()
    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(combined_categories)))

    all_avg_frequencies = []
    all_avg_magnitudes = []
    total_plotted_audios = 0

    for idx, combined_category in enumerate(combined_categories):
        subset = csv_data[csv_data["combined_category"] == combined_category]
        all_frequencies = []
        all_magnitudes = []

        for _, row in subset.iterrows():
            if total_plotted_audios >= num_audios_to_plot:
                break

            audio_file = row["nombre"]
            audio_path = os.path.join(audio_dir, audio_file)

            if not os.path.exists(audio_path):
                missing_files.add(audio_file)
                continue

            y, sr = librosa.load(audio_path)
            audio_duration = librosa.get_duration(y=y, sr=sr)
            if audio_duration < segment[1]:
                short_audios.append(audio_file)
                continue

            total_plotted_audios += 1

            start_sample = int(segment[0] * sr)
            end_sample = int(segment[1] * sr)
            y = y[start_sample:end_sample]

            fft_result = np.fft.fft(y)
            frequencies = np.fft.fftfreq(len(fft_result), 1 / sr)
            magnitude = np.abs(fft_result)

            positive_frequencies = frequencies[: len(frequencies) // 2]
            positive_magnitude_db = 20 * np.log10(
                magnitude[: len(magnitude) // 2] + 1e-9
            )

            all_frequencies.append(positive_frequencies)
            all_magnitudes.append(positive_magnitude_db)

            plt.plot(
                positive_frequencies,
                positive_magnitude_db,
                color=colors[idx],
                linewidth=0.1,
                alpha=0.002,
            )

        if all_frequencies:
            min_length = min(len(f) for f in all_frequencies)
            all_frequencies = [f[:min_length] for f in all_frequencies]
            all_magnitudes = [m[:min_length] for m in all_magnitudes]

            avg_frequencies = np.mean(all_frequencies, axis=0)
            avg_magnitude_db = np.mean(all_magnitudes, axis=0)

            all_avg_frequencies.append(avg_frequencies)
            all_avg_magnitudes.append(avg_magnitude_db)

    for idx, (avg_frequencies, avg_magnitude_db) in enumerate(
        zip(all_avg_frequencies, all_avg_magnitudes)
    ):
        smoothed_magnitude_db = moving_average(avg_magnitude_db, window_size)
        plt.plot(
            avg_frequencies[: len(smoothed_magnitude_db)],
            smoothed_magnitude_db,
            label=f"{combined_categories[idx]}",
            color=colors[idx],
            linewidth=1.5,
        )

    plt.title(title)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud (dB)")
    plt.ylim(bottom=-60, top=None)
    plt.grid(True)
    plt.xlim(20, None)
    plt.xticks(
        [20] + list(np.arange(500, max(avg_frequencies), step=500)),
        rotation=45,
    )
    plt.legend(loc="upper right", fontsize="small")

    plt.annotate(
        f"Directorio: {audio_dir}\nAudios graficados: {total_plotted_audios}"
        f"\nSegmento: {segment[1] - segment[0]} s ({segment[0]}-{segment[1]} s)",
        xy=(0.02, 0.05),
        xycoords="axes fraction",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"
        ),
    )

    plt.tight_layout()
    plt.show()

    if show_errors:
        if missing_files:
            print(f"Audios ausentes: {len(missing_files)}")
            for archivo in missing_files:
                print(f"Archivo no encontrado: {archivo}")

        if short_audios:
            print(f"Audios demasiado cortos: {len(short_audios)}")
            for audio in short_audios:
                print(f"Longitud insuficiente: {audio}")


def count_audio_combinations(
    csv_file: str, category_columns: list[str] = None
) -> None:
    """
    Cuenta la cantidad de audios por combinaciones de categorías especificadas.

    Args:
        csv_file (str): Ruta al archivo CSV que contiene los datos.
        category_columns (list[str], optional): Lista de nombres de las columnas a agrupar para contar la cantidad de audios.

    Raises:
        ValueError: Si alguna de las columnas especificadas no existe en el archivo CSV.
    """
    # Leer el archivo CSV
    df = pd.read_csv(csv_file)

    if category_columns is None:
        for column in df.columns:
            if (
                column != "nombre"
            ):  # Suponiendo que 'nombre' es la columna de archivo de audio
                count_df = (
                    df.groupby(column)
                    .size()
                    .reset_index(name="cantidad_audios")
                )
                print(f"Cantidad de audios por categoría '{column}':")
                print(count_df)
                print("\n")
    else:
        for column in category_columns:
            if column not in df.columns:
                raise ValueError(
                    f"La columna '{column}' no existe en el archivo CSV"
                )

        count_df = (
            df.groupby(category_columns)
            .size()
            .reset_index(name="cantidad_audios")
        )
        print(count_df)
