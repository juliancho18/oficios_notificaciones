import pandas as pd
import os


class Loaders:

    def cargar_excel(self, path_archivo: str, nombre_hoja: str):
        """
        Carga una hoja específica desde un archivo Excel.

        Args:
            path_archivo (str): Ruta completa del archivo Excel.
            nombre_hoja (str): Nombre de la hoja a cargar.

        Returns:
            pd.DataFrame | None
        """
        try:
            df = pd.read_excel(path_archivo, sheet_name=nombre_hoja)
            return df
        except Exception as e:
            print(f"❌ Error al leer el archivo Excel: {e}")
            return None

    def cargar_csv(self, path_archivo: str, sep: str = ",", encoding: str = "latin-1", on_bad_lines: str = "skip"):
        """
        Carga un CSV en un DataFrame, omitiendo líneas mal formadas si aplica.
        """
        try:
            return pd.read_csv(
                path_archivo,
                sep=sep,
                encoding=encoding,
                engine="python",
                on_bad_lines=on_bad_lines
            )
        except Exception as e:
            print(f"❌ Error al leer el archivo CSV: {e}")
            return None

    def guardar_excel(self, df: pd.DataFrame, nombre_archivo: str, carpeta_salida: str = "output"):
        """
        Guarda un DataFrame como archivo Excel en una carpeta de salida.
        """
        os.makedirs(carpeta_salida, exist_ok=True)
        ruta = os.path.join(os.getcwd(), carpeta_salida, f"{nombre_archivo}.xlsx")
        try:
            df.to_excel(ruta, index=False)
            print(f"✅ Archivo guardado exitosamente en: {ruta}")
        except Exception as e:
            print(f"❌ Error al guardar el archivo: {e}")