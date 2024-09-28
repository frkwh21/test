import streamlit as st
import pandas as pd
import pulp
import matplotlib.pyplot as plt
import japanize_matplotlib
import io

class ShiftScheduler:
    def __init__(self):
        self.S = []  # スタッフのリスト
        self.D = []  # 日付のリスト
        self.SD = []  # スタッフと日付の組のリスト
        self.S2leader_flag = {}  # スタッフの責任者フラグ
        self.S2min_shift = {}  # スタッフの希望最小出勤日数
        self.S2max_shift = {}  # スタッフの希望最大出勤日数
        self.D2required_staff = {}  # 各日の必要人数
        self.D2required_leader = {}  # 各日の必要責任者数
        self.x = {}  # 各スタッフが各日にシフトに入るか否かを表す変数
        self.y_under = {}  # 各スタッフの希望勤務日数の不足数を表すスラック変数
        self.y_over = {}  # 各スタッフの希望勤務日数の超過数を表すスラック変数
        self.model = None
        self.status = -1
        self.sch_df = None

    def set_data(self, staff_df, calendar_df):
        self.S = staff_df["スタッフID"].tolist()
        self.D = calendar_df["日付"].tolist()
        self.SD = [(s, d) for s in self.S for d in self.D]
        S2Dic = staff_df.set_index("スタッフID").to_dict()
        self.S2leader_flag = S2Dic["責任者フラグ"]
        self.S2min_shift = S2Dic["希望最小出勤日数"]
        self.S2max_shift = S2Dic["希望最大出勤日数"]
        D2Dic = calendar_df.set_index("日付").to_dict()
        self.D2required_staff = D2Dic["出勤人数"]
        self.D2required_leader = D2Dic["責任者人数"]

    def build_model(self):
        self.model = pulp.LpProblem("ShiftScheduler", pulp.LpMinimize)
        self.x = pulp.LpVariable.dicts("x", self.SD, cat="Binary")
        self.y_under = pulp.LpVariable.dicts("y_under", self.S, cat="Continuous", lowBound=0)
        self.y_over = pulp.LpVariable.dicts("y_over", self.S, cat="Continuous", lowBound=0)

        for d in self.D:
            self.model += pulp.lpSum(self.x[s, d] for s in self.S) >= self.D2required_staff[d]
            self.model += pulp.lpSum(self.x[s, d] * self.S2leader_flag[s] for s in self.S) >= self.D2required_leader[d]

        self.model += pulp.lpSum([self.y_under[s] for s in self.S]) + pulp.lpSum([self.y_over[s] for s in self.S])

        for s in self.S:
            self.model += self.S2min_shift[s] - pulp.lpSum(self.x[s,d] for d in self.D) <= self.y_under[s]
            self.model += pulp.lpSum(self.x[s,d] for d in self.D) - self.S2max_shift[s] <= self.y_over[s]

    def solve(self):
        solver = pulp.PULP_CBC_CMD(msg=0)
        self.status = self.model.solve(solver)
        Rows = [[int(self.x[s,d].value()) for d in self.D] for s in self.S]
        self.sch_df = pd.DataFrame(Rows, index=self.S, columns=self.D)
        return pulp.LpStatus[self.status], self.model.objective.value(), self.sch_df

def load_csv(uploaded_file):
    if uploaded_file is not None:
        try:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)
            return df
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
            return None
    return None

# Streamlitアプリケーション
st.title("シフトスケジューリングアプリ")

# サイドバー
st.sidebar.header("データのアップロード")
calendar_file = st.sidebar.file_uploader("カレンダー", type=["csv"])
staff_file = st.sidebar.file_uploader("スタッフ", type=["csv"])

# タブ
tab1, tab2, tab3 = st.tabs(["カレンダー情報", "スタッフ情報", "シフト表作成"])

with tab1:
    st.markdown("## カレンダー情報")
    calendar_df = load_csv(calendar_file)
    if calendar_df is not None:
        st.write(calendar_df)
    else:
        st.write("カレンダー情報をアップロードしてください")

with tab2:
    st.markdown("## スタッフ情報")
    staff_df = load_csv(staff_file)
    if staff_df is not None:
        st.write(staff_df)
    else:
        st.write("スタッフ情報をアップロードしてください")

with tab3:
    st.markdown("## シフト表作成")
    if calendar_df is not None and staff_df is not None:
        if st.button("最適化実行"):
            scheduler = ShiftScheduler()
            scheduler.set_data(staff_df, calendar_df)
            scheduler.build_model()
            status, objective_value, shift_table = scheduler.solve()
            
            st.markdown("### 最適化結果")
            st.write(f"ステータス: {status}")
            st.write(f"目的関数値: {objective_value}")
            
            st.markdown("### シフト表")
            st.write(shift_table)
            
            st.markdown("### シフト数の充足確認")
            required_staff = calendar_df.set_index("日付")["出勤人数"]
            actual_staff = shift_table.sum()
            comparison = pd.DataFrame({"必要人数": required_staff, "実際の人数": actual_staff})
            st.write(comparison)
            
            st.markdown("### スタッフの希望の確認")
            staff_preferences = staff_df.set_index("スタッフID")[["希望最小出勤日数", "希望最大出勤日数"]]
            actual_shifts = shift_table.sum(axis=1)
            preferences_comparison = pd.DataFrame({
                "希望最小出勤日数": staff_preferences["希望最小出勤日数"],
                "希望最大出勤日数": staff_preferences["希望最大出勤日数"],
                "実際の出勤日数": actual_shifts
            })
            st.write(preferences_comparison)
            
            st.markdown("### 責任者の合計シフト数の充足確認")
            required_leaders = calendar_df.set_index("日付")["責任者人数"]
            leader_flags = staff_df.set_index("スタッフID")["責任者フラグ"]
            actual_leaders = shift_table.apply(lambda col: (col * leader_flags).sum())
            leader_comparison = pd.DataFrame({"必要責任者数": required_leaders, "実際の責任者数": actual_leaders})
            st.write(leader_comparison)

            # シフト表の可視化
            st.markdown("### シフト表の可視化")
            fig, ax = plt.subplots(figsize=(12, 6))
            shift_sum = shift_table.sum(axis=1)  # 各日のシフト合計を計算
            shift_sum.plot(kind='bar', ax=ax)
            ax.set_xlabel('日付')
            ax.set_ylabel('シフト数')
            ax.set_title('日ごとのシフト割り当て数')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

            # 日ごとの総シフト数の可視化
            st.markdown("### 日ごとの総シフト数")
            fig, ax = plt.subplots(figsize=(12, 6))
            total_shifts = shift_table.sum()
            required_staff = calendar_df.set_index('日付')['出勤人数']
            
            x = range(len(total_shifts))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], total_shifts, width, label='実際のシフト数')
            ax.bar([i + width/2 for i in x], required_staff, width, label='必要人数')
            
            ax.set_xlabel('日付')
            ax.set_ylabel('人数')
            ax.set_title('日ごとの総シフト数と必要人数の比較')
            ax.set_xticks(x)
            ax.set_xticklabels(total_shifts.index, rotation=45)
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)

            # スタッフごとの総シフト数の可視化
            st.markdown("### スタッフごとの総シフト数")
            fig, ax = plt.subplots(figsize=(12, 6))
            staff_total_shifts = shift_table.sum(axis=1)
            staff_min_shifts = staff_df.set_index('スタッフID')['希望最小出勤日数']
            staff_max_shifts = staff_df.set_index('スタッフID')['希望最大出勤日数']
            
            x = range(len(staff_total_shifts))
            width = 0.2
            
            ax.bar([i - width for i in x], staff_min_shifts, width, label='希望最小日数')
            ax.bar([i for i in x], staff_total_shifts, width, label='実際のシフト数')
            ax.bar([i + width for i in x], staff_max_shifts, width, label='希望最大日数')
            
            ax.set_xlabel('スタッフ')
            ax.set_ylabel('シフト数')
            ax.set_title('スタッフごとの総シフト数と希望日数の比較')
            ax.set_xticks(x)
            ax.set_xticklabels(staff_total_shifts.index)
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)

            # 責任者の合計シフト数の可視化
            st.markdown("### 責任者の合計シフト数")
            fig, ax = plt.subplots(figsize=(12, 6))
            required_leaders = calendar_df.set_index('日付')['責任者人数']
            leader_flags = staff_df.set_index('スタッフID')['責任者フラグ']
            actual_leaders = shift_table.apply(lambda col: (col * leader_flags).sum())
            
            x = range(len(required_leaders))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], required_leaders, width, label='必要責任者数')
            ax.bar([i + width/2 for i in x], actual_leaders, width, label='実際の責任者数')
            
            ax.set_xlabel('日付')
            ax.set_ylabel('責任者数')
            ax.set_title('日ごとの必要責任者数と実際の責任者数の比較')
            ax.set_xticks(x)
            ax.set_xticklabels(required_leaders.index, rotation=45)
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.write("カレンダー情報とスタッフ情報の両方をアップロードしてから最適化を実行してください")
