import streamlit as st
import tempfile

from segmentation_core import analyze_image
from main18_logic import abcd_analysis
from evolution import evolution_score
from history import add_record


st.set_page_config(
    page_title="Аналіз утворень на шкірі",
    layout="centered"
)

st.title("🩺 SkinCare - аналіз утворень на шкірі ")
st.caption("Система аналізу та моніторингу шкірних утворень\nза правилом ABCDE")

st.markdown("---")


user_id = st.text_input("👤 Імʼя користувача", value="user_01")
lesion_id = st.text_input("🔬 Ідентифікатор утворення", value="lesion_01")

uploaded = st.file_uploader(
    "📷 Завантажте фото утворення",
    type=["jpg", "jpeg", "png"]
)
st.info(
        "📌 **Вимоги до фотографії:**\n"
        "- Зображення має бути зроблене при **однорідному світлі** без тіней.\n"
        "- Фото повинно містити **монету 10 гривень**, розташовану біля утворення — це потрібно для масштабування.\n"
        "- Зображення має бути **чітким**, у фокусі.\n"
        "- Фотографуйте **перпендикулярно до шкіри**."
    )


if st.button("🔍 Проаналізувати", use_container_width=True):

    if uploaded is None:
        st.warning("Будь ласка, завантажте зображення !")
        st.stop()



    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded.read())
        image_path = tmp.name

    with st.spinner("🔄 Аналіз зображення..."):
        result = analyze_image(image_path)

    if result is None or result["area_mm2"] is None:
        st.error("❌ Не вдалося виконати аналіз\n"
                 "Перевірте чи усі вимоги до фотографії були дотримані")

        st.stop()


    abcd = abcd_analysis(
        result["original"],
        result["final_mask"],
        result["area_mm2"]
    )


    add_record(
        user_id=user_id,
        lesion_id=lesion_id,
        area_mm2=result["area_mm2"],
        diameter_mm=abcd["D"],
        A=abcd["A"],
        B=abcd["B"],
        C=abcd["C"],
        risk=abcd["risk_abcd"]
    )

    evo = evolution_score(user_id, lesion_id)
    total_risk = abcd["risk_abcd"] + evo["E"] * 3.0



    st.markdown("## 📊 Результати аналізу")

    col1, col2 = st.columns(2)

    with col1:
        st.image(result["overlay"], caption="Сегментація утворення", channels="BGR")

    with col2:
        st.metric("📐 Площа", f"{result['area_mm2']:.2f} мм²")
        st.metric("D – Діаметр", f"{abcd['D']:.2f} мм")

        st.write("**ABCD:**")
        st.write(f"A — Асиметрія: {abcd['A']:.2f}")
        st.write(f"B — Край: {abcd['B']:.2f}")
        st.write(f"C — Колір: {abcd['C']:.2f}")

    st.markdown("---")

    st.write("📈 **Еволюція (E):**")
    st.json(evo)

    st.markdown("---")


    st.markdown("## ⚠️ Підсумковий ризик")

    if total_risk < 3:
        st.success(f"✅ Низький ризик ({total_risk:.2f})")
    elif total_risk < 6:
        st.warning(f"⚠️ Помірний ризик ({total_risk:.2f})")
    else:
        st.error(f"❌ Високий ризик ({total_risk:.2f})")

    st.caption(
        "⚠️ Система має дослідницький характер і не є медичним діагнозом"
    )
