import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator


def plot_data_by_month(filepath, building_name, year, X_train, X_val):
    # 데이터 불러오기
    df = pd.read_csv(filepath)
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    df = df.sort_values('Date/Time').reset_index(drop=True)

    # train/valid/test 경계 인덱스 계산
    train_end = X_train.shape[0]
    valid_end = train_end + X_val.shape[0]

    # 시각화
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 왼쪽 y축: electricity (파랑)
    elec_line, = ax1.plot(df['Date/Time'], df['electricity'],
                          color='tab:blue', label='Electricity')
    ax1.set_ylabel('Electricity', color='tab:blue')

    # 오른쪽 y축: Temp (빨강) / Rain (노랑)
    ax2 = ax1.twinx()
    temp_line, = ax2.plot(df['Date/Time'], df['Mean Temp (°C)'],
                          color='tab:red', label='Mean Temp (°C)')
    rain_line, = ax2.plot(df['Date/Time'], df['Total Rain (mm)'],
                          color='tab:orange', label='Total Rain (mm)')
    ax2.set_ylabel('Temperature / Rain', color='tab:red')

    # 경계선 표시 (---)
    split_line1 = ax1.axvline(df['Date/Time'].iloc[train_end],
                              color='gray', linestyle='--', label='Train/Valid Split')
    split_line2 = ax1.axvline(df['Date/Time'].iloc[valid_end],
                              color='black', linestyle='--', label='Valid/Test Split')

    # x축 라벨은 month만 표시
    ax1.xaxis.set_major_locator(MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(DateFormatter('%m'))
    ax1.set_xlabel('Month')


    # 범례 생성: 모든 라인을 합쳐서 표시
    lines = [elec_line, temp_line, rain_line, split_line1, split_line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    # 제목 표시
    title_str = f"{building_name} Electricity / Temperature / Rain Data in {year}"
    plt.title(title_str)

    plt.tight_layout()
    plt.show()

    # 저장
    fig.savefig(f"plots/{building_name}_{year}_data_plot.png", dpi=300)
    print(f"[저장 완료] plots/png/{building_name}_{year}_data_plot.png")
