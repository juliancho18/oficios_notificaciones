import pandas as pd
import os
import re
import unicodedata


class Utils:
    # =========================
    # Helpers de limpieza
    # =========================
    def normalizar_texto(self, s: str) -> str:
        """
        Normaliza texto: strip, espacios múltiples, conserva tildes (no las elimina).
        """
        if s is None:
            return ""
        s = str(s).strip()
        s = re.sub(r"\s+", " ", s)
        return s

    def _norm_txt(self, x: str) -> str:
        """
        Normaliza para llaves (regionales / matching):
        - MAYÚSCULAS
        - sin tildes
        - sin dobles espacios
        """
        x = "" if x is None else str(x)
        x = x.strip().upper()
        x = " ".join(x.split())
        x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
        return x

    def limpiar_nombre_archivo(self, s: str, max_len: int = 80) -> str:
        """
        Convierte un nombre (persona) en un nombre de archivo seguro para Windows.
        """
        s = self.normalizar_texto(s)
        if s == "":
            s = "SIN_NOMBRE"

        # Quitar caracteres no válidos en nombres de archivo
        s = re.sub(r'[\\/*?:"<>|]', "", s)

        # Quitar caracteres raros no imprimibles
        s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

        # Reemplazar espacios por guion bajo
        s = s.replace(" ", "_")

        # Limitar longitud
        return s[:max_len].rstrip("_")

    # =========================
    # Funciones base (las que ya tenías)
    # =========================
    def filtrar_columnas(self, df, vector):
        """
        Filtra el DataFrame para quedarse únicamente con las columnas indicadas en `vector`.
        """
        cols_existentes = [c for c in vector if c in df.columns]
        faltantes = [c for c in vector if c not in df.columns]

        if faltantes:
            print(f"⚠️ Columnas no encontradas y omitidas: {faltantes}")

        return df[cols_existentes].copy()

    def fusionar_dataframes(self, df1, df2, on, how="left"):
        """
        Fusiona dos DataFrames por una llave (o llaves).
        """
        return df1.merge(df2, on=on, how=how)

    def agregar_dummies(self, df, col):
        """
        Crea variables dummies (one-hot) para una columna categórica.
        """
        df_out = df.copy()
        dummies = pd.get_dummies(df_out[col], prefix=col, dummy_na=False)
        return pd.concat([df_out.drop(columns=[col]), dummies], axis=1)

    def generar_features_fecha(self, df, col_fecha):
        """
        Genera features a partir de una columna fecha: año, mes, día, etc.
        """
        df_out = df.copy()
        df_out[col_fecha] = pd.to_datetime(df_out[col_fecha], errors="coerce")
        df_out[f"{col_fecha}_anio"] = df_out[col_fecha].dt.year
        df_out[f"{col_fecha}_mes"] = df_out[col_fecha].dt.month
        df_out[f"{col_fecha}_dia"] = df_out[col_fecha].dt.day
        df_out[f"{col_fecha}_dia_semana"] = df_out[col_fecha].dt.day_name()
        return df_out

    def agregar_columna_binaria(self, df, col, valor_objetivo, nueva_columna):
        """
        Agrega una columna binaria: 1 si df[col] == valor_objetivo, si no 0.
        """
        df_out = df.copy()
        df_out[nueva_columna] = (df_out[col] == valor_objetivo).astype(int)
        return df_out

    def asignar_grupos(self, df, col):
        """
        Ejemplo de agrupación por lista fija (tu lógica).
        """
        df_out = df.copy()
        grupo1 = {"jeniffer caballero", "cristian gil", "valentina bernal"}

        df_out["grupo"] = (
            df_out[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .apply(lambda x: 1 if x in grupo1 else 2)
        )
        return df_out

    # =========================
    # Regional -> indicador, director, dirección, municipio (NO vacíos)
    # =========================
    def agregar_info_regional(self, df: pd.DataFrame, col_regional: str = "Regional") -> pd.DataFrame:
        """
        Agrega columnas según la Regional (NO se permite vacío):
          - INDICADOR
          - DIRECTOR_REGIONAL
          - DIRECCION_REGIONAL
          - MUNICIPIO_REGIONAL
        """
        if col_regional not in df.columns:
            raise KeyError(
                f"No existe la columna '{col_regional}' en el DataFrame.\n"
                f"Columnas disponibles: {df.columns.tolist()}"
            )

        regional_map = {
            self._norm_txt("ALMEIDAS Y GUATAVITA"): (
                "DRAG", "HECTOR JOSE CAMACHO ACOSTA",
                "Carrera. 5 N° 5-73 piso 2 y 4 Edificio Molino del Parque",
                "Chocontá, Cundinamarca",
            ),
            self._norm_txt("ALTO MAGDALENA"): (
                "DRAM", "CAMILA ANDREA VELASQUEZ BAQUERO",
                "Calle 21 No. 8 – 23 Barrio Granada",
                "Girardot, Cundinamarca",
            ),
            self._norm_txt("BAJO MAGDALENA"): (
                "DRBM", "JUAN FILIBERTO COTRINO GUEVARA",
                "Calle 4 N° 5-68 Camellón Real",
                "Guaduas, Cundinamarca",
            ),
            self._norm_txt("CHIQUINQUIRA"): (
                "DRCH", "YIBER ESTEBAN GONZALEZ GIL",
                "Carrera 6 N° 9-40",
                "Chiquinquirá, Boyacá",
            ),
            self._norm_txt("GUALIVA"): (
                "DRGU", "RONALD DAVID PRIETO GONZALEZ",
                "Calle. 8 N° 10ª - 41 Barrio El Recreo",
                "Villeta, Cundinamarca",
            ),
            self._norm_txt("MAGDALENA CENTRO"): (
                "DRMC", "JUAN CARLOS ESCOBAR CRISTANCHO",
                "Carrera 5 N° 3 - 02 Esquina",
                "Vianí, Cundinamarca",
            ),
            self._norm_txt("RIONEGRO"): (
                "DRRN", "ALEJANDRO FUQUITVA CASALLAS",
                "Calle 9 N° 19-72, Barrio Antonio Nariño",
                "Pacho, Cundinamarca",
            ),
            self._norm_txt("SABANA CENTRO"): (
                "DRSC", "ANDRES MAURICIO GARZON ORJUELA",
                "Calle 7 A N°11- 40 Barrio Algarra",
                "Zipaquirá, Cundinamarca",
            ),
            self._norm_txt("SABANA OCCIDENTE"): (
                "DRSO", "LINA CAMILA CORTES ACOSTA",
                "Carrera 4. N° 4-38",
                "Facatativá, Cundinamarca",
            ),
            self._norm_txt("SOACHA"): (
                "DRSOA", "CESAR AUGUSTO RICO MAYORGA",
                "Trans. 7 F N° 26-38 Barrio El Nogal - Soacha",
                "Soacha, Cundinamarca",
            ),
            self._norm_txt("SUMAPAZ"): (
                "DRSU", "ERIKA ALVAREZ CASTAÑEDA",
                "Avenida las Palmas N° 15-17",
                "Fusagasugá, Cundinamarca",
            ),
            self._norm_txt("TEQUENDAMA"): (
                "DRTE", "NIDIA CRUZ ORTEGA",
                "Carrera 21 con Calle 2da, Esquina Barrio El Recreo",
                "La Mesa, Cundinamarca",
            ),
            self._norm_txt("UBATE"): (
                "DRUB", "JULIO CESAR SIERRA LEON",
                "Transversal 2da. N°1E-40",
                "Ubaté, Cundinamarca",
            ),
            self._norm_txt("BOGOTA LA CALERA"): (
                "DRBC", "YUBER YESID CARDENAS PULIDO",
                "Carrera 10 N° 16-82 piso 4",
                "Bogotá, Cundinamarca",
            ),
        }

        df_out = df.copy()
        df_out["_regional_norm"] = df_out[col_regional].map(self._norm_txt)

        df_out["INDICADOR"] = df_out["_regional_norm"].map(lambda k: regional_map.get(k, (None, None, None, None))[0])
        df_out["DIRECTOR_REGIONAL"] = df_out["_regional_norm"].map(lambda k: regional_map.get(k, (None, None, None, None))[1])
        df_out["DIRECCION_REGIONAL"] = df_out["_regional_norm"].map(lambda k: regional_map.get(k, (None, None, None, None))[2])
        df_out["MUNICIPIO_REGIONAL"] = df_out["_regional_norm"].map(lambda k: regional_map.get(k, (None, None, None, None))[3])

        faltan = df_out[
            df_out["INDICADOR"].isna()
            | df_out["DIRECTOR_REGIONAL"].isna()
            | df_out["DIRECCION_REGIONAL"].isna()
            | df_out["MUNICIPIO_REGIONAL"].isna()
        ][col_regional].astype(str).unique().tolist()

        df_out.drop(columns=["_regional_norm"], inplace=True)

        if faltan:
            raise ValueError(
                "❌ Hay Regional(es) sin mapeo completo (INDICADOR/DIRECTOR/DIRECCION/MUNICIPIO):\n"
                + "\n".join([f"- {x}" for x in faltan])
                + "\n\nSolución: agrega esas regionales al diccionario regional_map."
            )

        return df_out

    # =========================
    # Cruce de correspondencia (mantengo ambas lógicas)
    # =========================
    def agregar_fila_cruce_excel(self, df, col_salida="No_Fila_Cruce", offset=2):
        """
        Numeración para referencia por fila en Excel (clásico):
        - Fila 1: encabezados
        - Fila 2: primer dato
        => offset=2
        """
        df_out = df.copy()
        df_out[col_salida] = range(offset, offset + len(df_out))
        return df_out

    def agregar_fila_cruce_word(self, df, col_salida="No_Fila_Cruce", offset=3):
        """
        Numeración para Cruce de correspondencia en Word cuando Word 'corre' 1 fila:
        - Fila 1 Excel: encabezados (Word lo toma como dato)
        - Fila 2 Excel: primer registro
        => necesitamos +1 adicional => offset=3
        """
        df_out = df.copy()
        df_out[col_salida] = range(offset, offset + len(df_out))
        return df_out

    def agregar_fila_cruce(self, df, col_salida="No_Fila_Cruce", offset=3):
        """
        Alias (por defecto) a la lógica que estás usando ahora: Word (offset=3).
        Si quieres Excel clásico, usa agregar_fila_cruce_excel().
        """
        return self.agregar_fila_cruce_word(df, col_salida=col_salida, offset=offset)

    # =========================
    # Exportar Excel por persona
    # =========================
    def exportar_excels_por_persona(
        self,
        df,
        col_persona,
        carpeta_salida="salidas_digitadores",
        nombre_prefix="BASE",
        add_fila_cruce=True,
        col_fila_cruce="No_Fila_Cruce",
        offset_fila_cruce=3,
    ):
        """
        Parte el DataFrame por col_persona y guarda un Excel por cada valor.
        """
        os.makedirs(carpeta_salida, exist_ok=True)

        df_work = df.copy()
        df_work[col_persona] = df_work[col_persona].astype(str).apply(self.normalizar_texto)

        resumen = []
        for persona, sub in df_work.groupby(col_persona, dropna=False):
            persona_str = self.normalizar_texto(persona)
            sub_out = sub.copy().reset_index(drop=True)

            if add_fila_cruce:
                sub_out = self.agregar_fila_cruce(sub_out, col_salida=col_fila_cruce, offset=offset_fila_cruce)

            safe_name = self.limpiar_nombre_archivo(persona_str)
            filename = f"{nombre_prefix}_{safe_name}.xlsx"
            ruta = os.path.join(carpeta_salida, filename)

            sub_out.to_excel(ruta, index=False)
            resumen.append((persona_str, len(sub_out), ruta))

        return pd.DataFrame(resumen, columns=["persona", "filas", "ruta_archivo"])

    # =========================
    # Partir DF por persona + cruce (sin perder datos)
    # =========================
    def partir_por_persona_con_cruce(
        self,
        df,
        col_persona="Notificador/Revisor",
        col_cruce="No_Fila_Cruce",
        offset_cruce=3,
    ):
        """
        Retorna un diccionario {persona: dataframe} sin perder datos,
        agregando columna de cruce (No_Fila_Cruce) en cada sub-dataframe.

        Importante:
        - Cada dataframe se resetea para que la numeración sea 100% consecutiva por persona.
        """
        if col_persona not in df.columns:
            raise KeyError(
                f"No existe la columna '{col_persona}' en el DataFrame.\n"
                f"Columnas disponibles: {df.columns.tolist()}"
            )

        df_work = df.copy()
        df_work[col_persona] = df_work[col_persona].astype(str).apply(self.normalizar_texto)

        salida = {}
        for persona, sub in df_work.groupby(col_persona, dropna=False):
            sub_out = sub.copy().reset_index(drop=True)
            sub_out = self.agregar_fila_cruce(sub_out, col_salida=col_cruce, offset=offset_cruce)
            salida[persona] = sub_out

        return salida