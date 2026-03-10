import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import pandas as pd


class Utils:

    # =========================
    # Helpers de limpieza
    # =========================
    def normalizar_texto(self, s: str) -> str:
        """
        Normaliza texto: strip, espacios múltiples, conserva tildes.
        """
        if s is None:
            return ""
        s = str(s).strip()
        s = re.sub(r"\s+", " ", s)
        return s

    def _norm_txt(self, x: str) -> str:
        """
        Normaliza para matching:
        - MAYÚSCULAS
        - sin tildes
        - sin dobles espacios
        """
        x = "" if x is None else str(x)
        x = x.strip().upper()
        x = " ".join(x.split())
        x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
        return x

    def limpiar_nombre_archivo(self, s: str, max_len: int = 120) -> str:
        """
        Convierte un texto en nombre de archivo seguro para Windows.
        """
        s = self.normalizar_texto(s)
        if s == "":
            s = "SIN_NOMBRE"

        s = re.sub(r'[\\/*?:"<>|]', "", s)
        s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
        s = s.replace(" ", "_")
        s = s[:max_len].rstrip("_")

        return s or "SIN_NOMBRE"

    def _ensure_dir(self, path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    def _safe(self, s: str, max_len: int = 120) -> str:
        return self.limpiar_nombre_archivo(s, max_len=max_len)

    # =========================
    # del / de la
    # =========================
    def _norm_lower_sin_tildes(self, x: str) -> str:
        x = "" if x is None else str(x)
        x = x.strip().lower()
        x = " ".join(x.split())
        x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
        return x

    def obtener_de_la_o_del(self, valor: str) -> str:
        """
        Devuelve 'de la' o 'del' según el sustantivo principal.
        """
        texto = self._norm_lower_sin_tildes(valor)
        if texto == "":
            return "de la"

        primera = texto.split()[0]

        femeninas = {
            "resolucion",
            "actuacion",
            "notificacion",
            "comunicacion",
            "liquidacion",
            "sancion",
            "certificacion",
        }
        masculinas = {
            "auto",
            "informe",
            "oficio",
            "acto",
            "memorando",
            "concepto",
        }

        if primera in femeninas:
            return "de la"
        if primera in masculinas:
            return "del"

        if primera.endswith("a"):
            return "de la"
        return "del"

    def agregar_columnas_del_de_la(
        self,
        df: pd.DataFrame,
        col_origen: str,
        col_articulo: str = "articulo_preposicion",
        col_frase: str = "descripcion_con_articulo",
        minusculas: bool = True,
    ) -> pd.DataFrame:
        """
        Agrega 2 columnas:
        1) col_articulo: 'de la' o 'del'
        2) col_frase: artículo + texto
        """
        if col_origen not in df.columns:
            raise KeyError(
                f"No existe la columna '{col_origen}' en el DataFrame.\n"
                f"Columnas disponibles: {df.columns.tolist()}"
            )

        out = df.copy()
        out[col_articulo] = out[col_origen].apply(self.obtener_de_la_o_del)

        if minusculas:
            base = out[col_origen].astype(str).str.strip().str.lower()
        else:
            base = out[col_origen].astype(str).str.strip()

        out[col_frase] = out[col_articulo] + " " + base
        return out

    # =========================
    # Funciones base
    # =========================
    def filtrar_columnas(self, df: pd.DataFrame, vector: list) -> pd.DataFrame:
        cols_existentes = [c for c in vector if c in df.columns]
        faltantes = [c for c in vector if c not in df.columns]

        if faltantes:
            print(f"⚠️ Columnas no encontradas y omitidas: {faltantes}")

        return df[cols_existentes].copy()

    def fusionar_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame, on, how: str = "left") -> pd.DataFrame:
        return df1.merge(df2, on=on, how=how)

    def agregar_dummies(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        df_out = df.copy()
        dummies = pd.get_dummies(df_out[col], prefix=col, dummy_na=False)
        return pd.concat([df_out.drop(columns=[col]), dummies], axis=1)

    def generar_features_fecha(self, df: pd.DataFrame, col_fecha: str) -> pd.DataFrame:
        df_out = df.copy()
        df_out[col_fecha] = pd.to_datetime(df_out[col_fecha], errors="coerce")
        df_out[f"{col_fecha}_anio"] = df_out[col_fecha].dt.year
        df_out[f"{col_fecha}_mes"] = df_out[col_fecha].dt.month
        df_out[f"{col_fecha}_dia"] = df_out[col_fecha].dt.day
        df_out[f"{col_fecha}_dia_semana"] = df_out[col_fecha].dt.day_name()
        return df_out

    def agregar_columna_binaria(self, df: pd.DataFrame, col: str, valor_objetivo, nueva_columna: str) -> pd.DataFrame:
        df_out = df.copy()
        df_out[nueva_columna] = (df_out[col] == valor_objetivo).astype(int)
        return df_out

    def asignar_grupos(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
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
    # Regional -> indicador, director, dirección, municipio
    # =========================
    def agregar_info_regional(self, df: pd.DataFrame, col_regional: str = "Regional") -> pd.DataFrame:
        if col_regional not in df.columns:
            raise KeyError(
                f"No existe la columna '{col_regional}' en el DataFrame.\n"
                f"Columnas disponibles: {df.columns.tolist()}"
            )

        regional_map: Dict[str, Tuple[str, str, str, str]] = {
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

        def _get_tuple(k: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
            return regional_map.get(k, (None, None, None, None))

        df_out["INDICADOR"] = df_out["_regional_norm"].map(lambda k: _get_tuple(k)[0])
        df_out["DIRECTOR_REGIONAL"] = df_out["_regional_norm"].map(lambda k: _get_tuple(k)[1])
        df_out["DIRECCION_REGIONAL"] = df_out["_regional_norm"].map(lambda k: _get_tuple(k)[2])
        df_out["MUNICIPIO_REGIONAL"] = df_out["_regional_norm"].map(lambda k: _get_tuple(k)[3])

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
    # Cruce de correspondencia
    # =========================
    def agregar_fila_cruce_excel(self, df: pd.DataFrame, col_salida: str = "No_Fila_Cruce", offset: int = 2) -> pd.DataFrame:
        df_out = df.copy()
        df_out[col_salida] = range(offset, offset + len(df_out))
        return df_out

    def agregar_fila_cruce_word(self, df: pd.DataFrame, col_salida: str = "No_Fila_Cruce", offset: int = 3) -> pd.DataFrame:
        df_out = df.copy()
        df_out[col_salida] = range(offset, offset + len(df_out))
        return df_out

    def agregar_fila_cruce(self, df: pd.DataFrame, col_salida: str = "No_Fila_Cruce", offset: int = 3) -> pd.DataFrame:
        return self.agregar_fila_cruce_word(df, col_salida=col_salida, offset=offset)

    # =========================
    # Exportar Excel por persona
    # =========================
    def exportar_excels_por_persona(
        self,
        df: pd.DataFrame,
        col_persona: str,
        carpeta_salida: str = "salidas_digitadores",
        nombre_prefix: str = "BASE",
        add_fila_cruce: bool = True,
        col_fila_cruce: str = "No_Fila_Cruce",
        offset_fila_cruce: int = 3,
    ) -> pd.DataFrame:
        os.makedirs(carpeta_salida, exist_ok=True)

        df_work = df.copy()
        df_work[col_persona] = df_work[col_persona].astype(str).apply(self.normalizar_texto)

        resumen = []
        for persona, sub in df_work.groupby(col_persona, dropna=False):
            persona_str = self.normalizar_texto(persona)
            sub_out = sub.copy().reset_index(drop=True)

            if add_fila_cruce:
                sub_out = self.agregar_fila_cruce(
                    sub_out,
                    col_salida=col_fila_cruce,
                    offset=offset_fila_cruce
                )

            safe_name = self.limpiar_nombre_archivo(persona_str)
            filename = f"{nombre_prefix}_{safe_name}.xlsx"
            ruta = os.path.join(carpeta_salida, filename)

            sub_out.to_excel(ruta, index=False)
            resumen.append((persona_str, len(sub_out), ruta))

        return pd.DataFrame(resumen, columns=["persona", "filas", "ruta_archivo"])

    # =========================
    # Partir DF por persona + cruce
    # =========================
    def partir_por_persona_con_cruce(
        self,
        df: pd.DataFrame,
        col_persona: str = "Notificador/Revisor",
        col_cruce: str = "No_Fila_Cruce",
        offset_cruce: int = 3,
    ) -> Dict[str, pd.DataFrame]:
        if col_persona not in df.columns:
            raise KeyError(
                f"No existe la columna '{col_persona}' en el DataFrame.\n"
                f"Columnas disponibles: {df.columns.tolist()}"
            )

        df_work = df.copy()
        df_work[col_persona] = df_work[col_persona].astype(str).apply(self.normalizar_texto)

        salida: Dict[str, pd.DataFrame] = {}
        for persona, sub in df_work.groupby(col_persona, dropna=False):
            sub_out = sub.copy().reset_index(drop=True)
            sub_out = self.agregar_fila_cruce(sub_out, col_salida=col_cruce, offset=offset_cruce)
            salida[str(persona)] = sub_out

        return salida

    # =========================
    # Lógica: ejecutorias / tipo oficio
    # =========================
    def _contiene_ejecutoria(self, x: str) -> bool:
        if x is None:
            return False

        txt = str(x).strip().lower()
        txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
        txt = " ".join(txt.split())

        patrones = [
            "ejecutoria",
            "para ejecutoria",
            "en ejecutoria",
            "ejecutoriado",
        ]

        return any(p in txt for p in patrones)

    def _norm_tipo_oficio(self, x: str) -> str:
        t = "" if x is None else str(x).strip().lower()
        t = " ".join(t.split())
        t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")

        if "cita" in t:
            return "citacion"
        if "comun" in t:
            return "comunicacion"
        if "electr" in t or "correo" in t:
            return "notificacion_electronica"
        if "aviso" in t:
            return "aviso"
        return t if t else "sin_tipo"

    # =========================
    # Excel operativo por abogado
    # =========================
    def exportar_excel_oficios_y_ejecutorias_por_notificador(
        self,
        df: pd.DataFrame,
        col_notificador: str = "Notificador/Revisor",
        col_estado_actuacion: str = "Estado_Actuacion",
        col_tipo_oficio: str = "Tipo_Oficio",
        col_expediente: str = "numero_expediente",
        col_notificado: str = "Nombre_Notificado",
        carpeta_salida: str = "output_abogados",
        nombre_archivo: str = "RESUMEN_OFICIOS_Y_EJECUTORIAS.xlsx",
        add_fila_cruce: bool = True,
        col_fila_cruce: str = "No_Fila_Cruce",
        offset_fila_cruce: int = 3,
    ) -> pd.DataFrame:
        for c in [col_notificador, col_estado_actuacion, col_tipo_oficio, col_expediente, col_notificado]:
            if c not in df.columns:
                raise KeyError(f"No existe la columna '{c}'. Columnas disponibles: {df.columns.tolist()}")

        self._ensure_dir(carpeta_salida)

        dfw = df.copy()
        dfw[col_notificador] = dfw[col_notificador].astype(str).apply(self.normalizar_texto)

        resumen = []

        for abogado, sub in dfw.groupby(col_notificador, dropna=False):
            abogado_txt = self.normalizar_texto(abogado)
            carpeta_abogado = self._ensure_dir(os.path.join(carpeta_salida, self._safe(abogado_txt)))

            mask_ej = (
                sub[col_estado_actuacion].apply(self._contiene_ejecutoria)
                | sub[col_tipo_oficio].apply(self._contiene_ejecutoria)
            )
            df_ej = sub[mask_ej].copy().reset_index(drop=True)
            df_of = sub[~mask_ej].copy().reset_index(drop=True)

            df_of_tab = df_of[[col_tipo_oficio, col_expediente, col_notificado, col_estado_actuacion]].copy()
            df_of_tab.rename(
                columns={
                    col_tipo_oficio: "tipo_oficio",
                    col_expediente: "numero_expediente",
                    col_notificado: "notificado",
                    col_estado_actuacion: "estado_actuacion",
                },
                inplace=True,
            )

            df_ej_tab = df_ej[[col_expediente, col_notificado, col_estado_actuacion]].copy()
            df_ej_tab.rename(
                columns={
                    col_expediente: "numero_expediente",
                    col_notificado: "notificado",
                    col_estado_actuacion: "estado_actuacion",
                },
                inplace=True,
            )

            if add_fila_cruce:
                df_of_tab = self.agregar_fila_cruce(df_of_tab, col_salida=col_fila_cruce, offset=offset_fila_cruce)
                df_ej_tab = self.agregar_fila_cruce(df_ej_tab, col_salida=col_fila_cruce, offset=offset_fila_cruce)

            ruta = os.path.join(carpeta_abogado, nombre_archivo)

            with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
                df_of_tab.to_excel(writer, index=False, sheet_name="OFICIOS_POR_CREAR")
                df_ej_tab.to_excel(writer, index=False, sheet_name="EJECUTORIAS")

            resumen.append(
                {
                    "abogado": abogado_txt,
                    "oficios_por_crear": len(df_of_tab),
                    "ejecutorias": len(df_ej_tab),
                    "ruta_excel": ruta,
                }
            )

        return pd.DataFrame(resumen)

    # =========================
    # Helpers Word desde plantillas externas
    # =========================
    def _ruta_base_proyecto(self) -> Path:
        return Path(__file__).resolve().parent

    def _ruta_plantillas(self) -> Path:
        ruta = self._ruta_base_proyecto() / "plantillas"
        ruta.mkdir(exist_ok=True)
        return ruta

    def _mapa_plantillas_docx(self) -> Dict[str, str]:
        return {
            "aviso": "Formato Aviso Recurso (AUTOMATIZADO).docx",
            "citacion": "Formato Citación CPACA2011 (Automatizado).docx",
            "comunicacion": "Formato Comunicación (Automatizado).docx",
            "notificacion_electronica": "Notificación electrónica (Automatizado).docx",
        }

    def _obtener_ruta_plantilla_por_tipo(self, tipo_oficio: str) -> str:
        tipo = self._norm_tipo_oficio(tipo_oficio)
        mapa = self._mapa_plantillas_docx()

        if tipo not in mapa:
            raise ValueError(
                f"Tipo de oficio no reconocido: {tipo_oficio}. "
                f"Tipos válidos: {list(mapa.keys())}"
            )

        ruta = self._ruta_plantillas() / mapa[tipo]

        if not ruta.exists():
            raise FileNotFoundError(f"No existe la plantilla: {ruta}")

        return str(ruta)

    def _reemplazar_texto_en_parrafo(self, parrafo, replacements: Dict[str, str]) -> None:
        texto_original = parrafo.text
        texto_nuevo = texto_original

        for buscar, reemplazar in replacements.items():
            if buscar in texto_nuevo:
                texto_nuevo = texto_nuevo.replace(buscar, reemplazar)

        if texto_nuevo == texto_original:
            return

        estilo = parrafo.runs[0] if parrafo.runs else None

        for run in parrafo.runs[::-1]:
            parrafo._element.remove(run._element)

        nuevo_run = parrafo.add_run(texto_nuevo)

        if estilo:
            nuevo_run.bold = estilo.bold
            nuevo_run.italic = estilo.italic
            nuevo_run.underline = estilo.underline
            nuevo_run.font.name = estilo.font.name
            nuevo_run.font.size = estilo.font.size

    def _replace_in_doc_generico(self, doc, replacements: Dict[str, str]) -> None:
        for p in doc.paragraphs:
            self._reemplazar_texto_en_parrafo(p, replacements)

        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        self._reemplazar_texto_en_parrafo(p, replacements)

        for section in doc.sections:
            for p in section.header.paragraphs:
                self._reemplazar_texto_en_parrafo(p, replacements)
            for p in section.footer.paragraphs:
                self._reemplazar_texto_en_parrafo(p, replacements)

            for table in section.header.tables:
                for row in table.rows:
                    for cell in row.cells:
                        for p in cell.paragraphs:
                            self._reemplazar_texto_en_parrafo(p, replacements)

            for table in section.footer.tables:
                for row in table.rows:
                    for cell in row.cells:
                        for p in cell.paragraphs:
                            self._reemplazar_texto_en_parrafo(p, replacements)

    def _armar_replacements_desde_row(
        self,
        row: pd.Series,
        columnas_df: List[str],
        col_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Arma reemplazos robustos para Word.
        Soporta:
        - «Campo»
        - <<Campo>>
        - [Campo]
        - {{Campo}}
        """
        if col_map is None:
            col_map = {}

        def _norm_key(x: str) -> str:
            x = "" if x is None else str(x)
            x = x.strip().lower()
            x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
            x = re.sub(r"[^a-z0-9]+", "", x)
            return x

        def _val(col_name: str) -> str:
            if col_name in row.index:
                v = row.get(col_name, "")
                if pd.isna(v):
                    return ""
                return str(v)
            return ""

        replacements: Dict[str, str] = {}

        for col in columnas_df:
            valor = _val(col)

            variantes = {
                str(col),
                str(col).strip(),
                _norm_key(col),
            }

            for k in variantes:
                replacements[f"«{k}»"] = valor
                replacements[f"<<{k}>>"] = valor
                replacements[f"[{k}]"] = valor
                replacements[f"{{{{{k}}}}}"] = valor

        for placeholder, real_col in col_map.items():
            valor = _val(real_col)

            variantes = {
                str(placeholder),
                str(placeholder).strip(),
                _norm_key(placeholder),
            }

            for k in variantes:
                replacements[f"«{k}»"] = valor
                replacements[f"<<{k}>>"] = valor
                replacements[f"[{k}]"] = valor
                replacements[f"{{{{{k}}}}}"] = valor

        return replacements

    # =========================
    # Generar .docx desde plantilla única
    # =========================
    def generar_words_desde_plantilla(
        self,
        df: pd.DataFrame,
        plantilla_path: str,
        carpeta_salida: str = "output_abogados",
        col_notificador: str = "Notificador/Revisor",
        col_subcarpeta: str = "ACTUACION",
        nombre_archivo_cols: Optional[List[str]] = None,
        col_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Genera un .docx por cada fila usando una plantilla .docx y reemplazando placeholders.
        """
        try:
            from docx import Document
        except Exception as e:
            raise ImportError(
                "No está instalada la librería python-docx. Instala con:\n"
                "pip install python-docx\n"
                f"Detalle: {e}"
            )

        if nombre_archivo_cols is None:
            nombre_archivo_cols = ["numero_expediente"]

        if col_map is None:
            col_map = {}

        if not os.path.exists(plantilla_path):
            raise FileNotFoundError(f"No existe la plantilla: {plantilla_path}")

        self._ensure_dir(carpeta_salida)

        generados = []

        for i, row in df.iterrows():
            abogado = self._safe(row.get(col_notificador, "SIN_ABOGADO"))
            sub = self._safe(row.get(col_subcarpeta, "SIN_TIPO")) if col_subcarpeta in df.columns else "SIN_TIPO"

            out_dir = os.path.join(carpeta_salida, abogado, "WORDS", sub)
            self._ensure_dir(out_dir)

            replacements = self._armar_replacements_desde_row(
                row=row,
                columnas_df=list(df.columns),
                col_map=col_map,
            )

            partes = []
            for c in nombre_archivo_cols:
                if c in df.columns:
                    partes.append(self._safe(row.get(c, "")))

            if not partes:
                partes = [f"documento_{i+1}"]

            nombre = "OFICIO_" + "_".join([p for p in partes if p]) + ".docx"
            ruta = os.path.join(out_dir, nombre)

            doc = Document(plantilla_path)
            self._replace_in_doc_generico(doc, replacements)
            doc.save(ruta)

            generados.append(
                {
                    "fila": i,
                    "ruta_docx": ruta,
                    "abogado": abogado,
                    "subcarpeta": sub,
                    "archivo": nombre,
                }
            )

        return pd.DataFrame(generados)

    # =========================
    # Generar .docx según tipo de oficio
    # =========================
    def generar_words_desde_tipo_oficio(
        self,
        df: pd.DataFrame,
        carpeta_salida: str = "output_abogados",
        col_notificador: str = "Notificador/Revisor",
        col_estado_actuacion: str = "Estado_Actuacion",
        col_tipo_oficio: str = "Tipo_Oficio",
        col_subcarpeta: Optional[str] = None,
        nombre_archivo_cols: Optional[List[str]] = None,
        col_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Genera un .docx por fila usando la plantilla correcta según Tipo_Oficio.
        Excluye ejecutorias tanto por estado como por actuación/tipo.
        """
        try:
            from docx import Document
        except Exception as e:
            raise ImportError(
                "No está instalada la librería python-docx. Instala con:\n"
                "pip install python-docx\n"
                f"Detalle: {e}"
            )

        if nombre_archivo_cols is None:
            nombre_archivo_cols = ["numero_expediente", "Nombre_Notificado"]

        if col_map is None:
            col_map = {}

        for c in [col_notificador, col_estado_actuacion, col_tipo_oficio]:
            if c not in df.columns:
                raise KeyError(
                    f"No existe la columna '{c}'. "
                    f"Columnas disponibles: {df.columns.tolist()}"
                )

        generados = []

        dfw = df.copy()

        mask_ej_estado = dfw[col_estado_actuacion].apply(self._contiene_ejecutoria)
        mask_ej_tipo = dfw[col_tipo_oficio].apply(self._contiene_ejecutoria)

        dfw = dfw[~(mask_ej_estado | mask_ej_tipo)].copy()

        for i, row in dfw.reset_index(drop=True).iterrows():
            abogado = self.normalizar_texto(row.get(col_notificador, "SIN_ABOGADO"))
            tipo_original = row.get(col_tipo_oficio, "")
            tipo = self._norm_tipo_oficio(tipo_original)

            if tipo not in {"aviso", "citacion", "comunicacion", "notificacion_electronica"}:
                generados.append(
                    {
                        "fila": i,
                        "abogado": abogado,
                        "tipo_oficio": tipo,
                        "error": f"Tipo no válido para generar entregable: {tipo_original}",
                    }
                )
                continue

            try:
                plantilla_path = self._obtener_ruta_plantilla_por_tipo(tipo)

                sub = self._safe(tipo)

                out_dir = os.path.join(carpeta_salida, self._safe(abogado), "WORDS", sub)
                self._ensure_dir(out_dir)

                replacements = self._armar_replacements_desde_row(
                    row=row,
                    columnas_df=list(dfw.columns),
                    col_map=col_map,
                )

                partes = []
                for c in nombre_archivo_cols:
                    if c in dfw.columns:
                        partes.append(self._safe(row.get(c, "")))

                if not partes:
                    partes = [f"documento_{i+1}"]

                nombre = "OFICIO_" + "_".join([p for p in partes if p]) + ".docx"
                ruta = os.path.join(out_dir, nombre)

                doc = Document(plantilla_path)
                self._replace_in_doc_generico(doc, replacements)
                doc.save(ruta)

                generados.append(
                    {
                        "fila": i,
                        "abogado": abogado,
                        "tipo_oficio": tipo,
                        "plantilla": os.path.basename(plantilla_path),
                        "ruta_docx": ruta,
                        "archivo": nombre,
                    }
                )

            except Exception as e:
                generados.append(
                    {
                        "fila": i,
                        "abogado": abogado,
                        "tipo_oficio": tipo,
                        "error": str(e),
                    }
                )

        return pd.DataFrame(generados)


def find_col(df: pd.DataFrame, posibles: List[str]) -> Optional[str]:
    def norm(s):
        return str(s).strip().lower().replace(" ", "").replace("_", "").replace(".", "")

    cols = list(df.columns)
    cols_norm = {norm(c): c for c in cols}

    for p in posibles:
        pn = norm(p)
        if pn in cols_norm:
            return cols_norm[pn]

    for p in posibles:
        pn = norm(p)
        for c in cols:
            if pn in norm(c):
                return c

    return None


def debug_detectar_columnas(df_filtrado: pd.DataFrame) -> Dict[str, Optional[str]]:
    print("Columnas DF (top 30):")
    print(list(df_filtrado.columns)[:30])

    col_notificador = find_col(df_filtrado, ["Notificador/Revisor", "Notificador", "Revisor"])
    col_estado = find_col(df_filtrado, ["Estado_Actuacion", "Estado Actuacion", "Estado", "Estado_Actuación", "Estado Actuación", "Estado de la Actuacion"])
    col_tipo_oficio = find_col(df_filtrado, ["Tipo_Oficio", "Tipo Oficio", "Tipo de oficio", "Tipo Oficio a generar", "ACTUACION"])
    col_expediente = find_col(df_filtrado, ["numero_expediente", "No. Exp.", "No Exp", "Expediente", "No_Expediente"])
    col_notificado = find_col(df_filtrado, ["Nombre_Notificado", "Nombre Notificado", "Notificado", "Nombre del Notificado"])

    print("\nDetectadas:")
    print("col_notificador:", col_notificador)
    print("col_estado     :", col_estado)
    print("col_tipo_oficio:", col_tipo_oficio)
    print("col_expediente :", col_expediente)
    print("col_notificado :", col_notificado)

    faltan = [
        k for k, v in {
            "col_notificador": col_notificador,
            "col_estado": col_estado,
            "col_tipo_oficio": col_tipo_oficio,
            "col_expediente": col_expediente,
            "col_notificado": col_notificado,
        }.items() if v is None
    ]

    if faltan:
        raise KeyError(f"❌ No pude detectar estas columnas: {faltan}. Revisa nombres en df_filtrado.columns.")

    return {
        "col_notificador": col_notificador,
        "col_estado": col_estado,
        "col_tipo_oficio": col_tipo_oficio,
        "col_expediente": col_expediente,
        "col_notificado": col_notificado,
    }