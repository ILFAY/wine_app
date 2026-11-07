import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent

st.set_page_config(page_title='Подбор хороших вин', page_icon='🍷', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_config():
    return json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
config = load_config()
test_df = load_test()

FEATURES = [c for c in test_df.columns if c != 'target']
NUMERIC_FEATURES = [c for c in FEATURES if c != 'is_red']
X_test = test_df[FEATURES]
y_test = test_df['target'].to_numpy()
proba_test = test_proba(model, X_test)

RU_NAMES = {
    'fixed_acidity': 'Нелетучая кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный SO2',
    'total_sulfur_dioxide': 'Общий SO2',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь, %',
    'is_red': 'Красное вино',
}


st.sidebar.header('Настройки')
threshold = st.sidebar.slider(
    'Порог отнесения к «хорошему вину»',
    min_value=0.0, max_value=1.0,
    value=float(config['threshold']), step=0.01,
    key='threshold',
    help='Вино попадает в подборку, если вероятность не ниже порога.',
)
st.sidebar.caption(f"Порог из config.json: {config['threshold']:.3f}")
st.sidebar.caption(f'Объектов в тесте: {len(test_df)}, из них хороших: {int(y_test.sum())}')

st.title('🍷 Подбор хороших вин')
st.caption('Логистическая регрессия по 12 физико-химическим характеристикам. '
           'Модель обучена в ноутбуке и здесь только применяется.')

tab_one, tab_metrics, tab_batch, tab_coefs = st.tabs(
    ['Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])


with tab_one:
    st.subheader('Оценка одного вина')
    st.write('Значения по умолчанию равны медиане по тестовой выборке, '
             'границы ползунков взяты из её минимума и максимума.')

    values = {}
    columns = st.columns(3)
    for i, feature in enumerate(NUMERIC_FEATURES):
        column = test_df[feature]
        low, high, median = float(column.min()), float(column.max()), float(column.median())
        with columns[i % 3]:
            values[feature] = st.number_input(
                RU_NAMES.get(feature, feature),
                min_value=low, max_value=high, value=median,
                step=(high - low) / 100, format='%.4f',
                key=f'input_{feature}',
            )

    wine_type = st.radio('Тип вина', ['Красное', 'Белое'], horizontal=True)
    values['is_red'] = 1 if wine_type == 'Красное' else 0

    one_wine = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    proba_one = float(model.predict_proba(one_wine)[0, 1])
    is_good = proba_one >= threshold

    left, right = st.columns(2)
    left.metric('Вероятность, что вино хорошее', f'{proba_one:.1%}')
    right.metric('Решение при текущем пороге',
                 'В подборку' if is_good else 'Отклонить',
                 delta=f'{proba_one - threshold:+.1%} к порогу')
    st.progress(min(proba_one, 1.0))

    if is_good:
        st.success(f'Вероятность {proba_one:.1%} не ниже порога {threshold:.2f}: '
                   f'вино попадает в подборку.')
    else:
        need = threshold - proba_one
        st.warning(f'Вероятность {proba_one:.1%} ниже порога {threshold:.2f} на '
                   f'{need:.1%}: вино в подборку не попадает.')


with tab_metrics:
    st.subheader('Модельные метрики на тестовой выборке')

    y_pred = (proba_test >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    precision = metrics.precision_score(y_test, y_pred, zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Precision', f'{precision:.3f}')
    c2.metric('Recall', f'{recall:.3f}')
    c3.metric('F1', f'{f1:.3f}')
    c4.metric('Вин в подборке', int(y_pred.sum()))

    left, right = st.columns([1, 1])

    with left:
        st.markdown('**Матрица ошибок**')
        confusion = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=['Факт: посредственное', 'Факт: хорошее'],
            columns=['Прогноз: отклонить', 'Прогноз: в подборку'])
        st.dataframe(confusion)
        st.caption(f'FP = {fp}: посредственное вино попало в подборку, клиент разочаруется. '
                   f'FN = {fn}: хорошее вино пропущено, подборка беднее.')

    with right:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.imshow([[tn, fp], [fn, tp]], cmap='Blues')
        for i, row in enumerate([[tn, fp], [fn, tp]]):
            for j, v in enumerate(row):
                ax.text(j, i, str(v), ha='center', va='center', fontsize=14)
        ax.set_xticks([0, 1], ['отклонить', 'в подборку'])
        ax.set_yticks([0, 1], ['посредственное', 'хорошее'])
        ax.set_xlabel('Прогноз')
        ax.set_ylabel('Факт')
        ax.grid(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    fpr, tpr, _ = metrics.roc_curve(y_test, proba_test)
    prec_curve, rec_curve, _ = metrics.precision_recall_curve(y_test, proba_test)
    roc_auc = metrics.roc_auc_score(y_test, proba_test)
    pr_auc = metrics.average_precision_score(y_test, proba_test)
    baseline = float(y_test.mean())
    current_fpr = fp / (fp + tn) if (fp + tn) else 0.0

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.4f}')
        ax.plot([0, 1], [0, 1], '--', color='gray', label='случайная модель')
        ax.scatter([current_fpr], [recall], s=80, color='crimson', zorder=5,
                   label=f'порог {threshold:.2f}')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR (recall)')
        ax.set_title('ROC-кривая')
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.plot(rec_curve, prec_curve, label=f'PR-AUC = {pr_auc:.4f}')
        ax.axhline(baseline, ls='--', color='gray', label=f'случайная модель = {baseline:.3f}')
        ax.scatter([recall], [precision], s=80, color='crimson', zorder=5,
                   label=f'порог {threshold:.2f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('PR-кривая')
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        f'**Что это значит для подборки.** При пороге {threshold:.2f} в подборку попадает '
        f'{int(y_pred.sum())} вин из {len(y_test)}, из них действительно хороших {tp}. '
        f'Это {recall:.0%} всех хороших вин теста, а доля удачных рекомендаций '
        f'{precision:.0%}.\n\n'
        'Если поднять порог, модель станет придирчивее: в подборке останется меньше вин, '
        'зато доля действительно хороших вырастет, а разочарованных клиентов станет меньше. '
        'Если опустить порог, подборка станет шире и мы потеряем меньше хороших вин, '
        'но вместе с ними наберём больше посредственных. '
        'Сдвиг порога не делает модель умнее, он только выбирает точку на этой же PR-кривой.'
    )


with tab_batch:
    st.subheader('Пакетное предсказание')
    st.write(f'Загрузите CSV с признаками вин (без столбца `target`). '
             f'Используется порог {threshold:.2f} из сайдбара.')
    st.caption('Нужны столбцы: ' + ', '.join(f'`{c}`' for c in FEATURES))

    uploaded = st.file_uploader('CSV-файл', type='csv')

    if uploaded is None:
        st.caption('Файл не загружен. Для проверки подойдёт test.csv из папки приложения, '
                   'столбец target будет проигнорирован.')
    else:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as error:
            st.error(f'Не удалось прочитать CSV: {error}')
        else:
            missing = [c for c in FEATURES if c not in batch.columns]
            if missing:
                st.error('В файле не хватает столбцов: ' + ', '.join(missing) +
                         '. Добавьте их и загрузите файл заново.')
            else:
                work = batch[FEATURES].copy()
                bad_rows = work.isna().any(axis=1).sum()
                if bad_rows:
                    st.warning(f'Строк с пропусками: {bad_rows}. Они исключены из расчёта.')
                    work = work.dropna()

                if work.empty:
                    st.error('После удаления строк с пропусками не осталось данных.')
                else:
                    proba_batch = model.predict_proba(work)[:, 1]
                    result = batch.loc[work.index].copy()
                    result['вероятность'] = proba_batch.round(4)
                    result['решение'] = np.where(proba_batch >= threshold,
                                                 'в подборку', 'отклонить')
                    result = result.sort_values('вероятность', ascending=False)

                    selected = int((proba_batch >= threshold).sum())
                    c1, c2, c3 = st.columns(3)
                    c1.metric('Вин в файле', len(work))
                    c2.metric('Попало в подборку', selected)
                    c3.metric('Доля подборки', f'{selected / len(work):.1%}')

                    st.dataframe(result, height=420)
                    st.download_button(
                        'Скачать результат в CSV',
                        data=result.to_csv(index=False).encode('utf-8-sig'),
                        file_name='wine_predictions.csv',
                        mime='text/csv')


with tab_coefs:
    st.subheader('Что выучила модель')
    st.write('Коэффициенты логистической регрессии после масштабирования признаков. '
             'Признаки приведены к одной шкале, поэтому величины сравнимы между собой.')

    classifier = model[-1]
    coefs = pd.Series(classifier.coef_.ravel(), index=FEATURES).sort_values()
    labels = [RU_NAMES.get(name, name) for name in coefs.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(labels, coefs.to_numpy(),
            color=['#c0564b' if v < 0 else '#3a7d44' for v in coefs])
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel('Коэффициент')
    ax.set_title('Вклад признаков в оценку «хорошее вино»')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    top_positive = coefs.sort_values(ascending=False).head(3)
    top_negative = coefs.head(3)

    left, right = st.columns(2)
    with left:
        st.markdown('**Сильнее всего голосуют за хорошее вино**')
        for name, value in top_positive.items():
            st.write(f'- {RU_NAMES.get(name, name)}: {value:+.3f}')
    with right:
        st.markdown('**Сильнее всего голосуют против**')
        for name, value in top_negative.items():
            st.write(f'- {RU_NAMES.get(name, name)}: {value:+.3f}')

    st.caption(
        'Знак коэффициента показывает направление: положительный увеличивает вероятность '
        'того, что вино хорошее, отрицательный уменьшает. Модуль показывает силу влияния '
        'при изменении признака на одно стандартное отклонение. Это зависимости, '
        'обнаруженные в данных, а не причинно-следственные связи: признаки коррелируют '
        'между собой, поэтому отдельный коэффициент нельзя читать как «добавь алкоголя и '
        'вино станет лучше».')
    st.dataframe(coefs.rename('коэффициент').to_frame().round(4))
