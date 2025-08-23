CREATE TABLE application
(
    sk_id_curr Int64,
    name_contract_type Nullable(String),
    code_gender Nullable(String),
    flag_own_car Nullable(String),
    flag_own_realty Nullable(String),
    cnt_children Nullable(Int32),
    amt_income_total Nullable(Float64),
    amt_credit Nullable(Float64),
    amt_annuity Nullable(Float64),
    amt_goods_price Nullable(Float64),
    name_type_suite Nullable(String),
    name_income_type Nullable(String),
    name_education_type Nullable(String),
    name_family_status Nullable(String),
    name_housing_type Nullable(String),
    region_population_relative Nullable(Float64),
    days_birth Nullable(Int32),
    days_employed Nullable(Int32),
    days_registration Nullable(Float64),
    days_id_publish Nullable(Int32),
    own_car_age Nullable(Float64),
    flag_mobil Nullable(Int32),
    flag_emp_phone Nullable(Int32),
    flag_work_phone Nullable(Int32),
    flag_cont_mobile Nullable(Int32),
    flag_phone Nullable(Int32),
    flag_email Nullable(Int32),
    occupation_type Nullable(String),
    cnt_fam_members Nullable(Float64),
    region_rating_client Nullable(Int32),
    region_rating_client_w_city Nullable(Int32),
    weekday_appr_process_start Nullable(String),
    hour_appr_process_start Nullable(Int32),
    reg_region_not_live_region Nullable(Int32),
    reg_region_not_work_region Nullable(Int32),
    live_region_not_work_region Nullable(Int32),
    reg_city_not_live_city Nullable(Int32),
    reg_city_not_work_city Nullable(Int32),
    live_city_not_work_city Nullable(Int32),
    organization_type Nullable(String),
    ext_source_1 Nullable(Float64),
    ext_source_2 Nullable(Float64),
    ext_source_3 Nullable(Float64),
    apartments_avg Nullable(Float64),
    basementarea_avg Nullable(Float64),
    years_beginexpluatation_avg Nullable(Float64),
    years_build_avg Nullable(Float64),
    commonarea_avg Nullable(Float64),
    elevators_avg Nullable(Float64),
    entrances_avg Nullable(Float64),
    floorsmax_avg Nullable(Float64),
    floorsmin_avg Nullable(Float64),
    landarea_avg Nullable(Float64),
    livingapartments_avg Nullable(Float64),
    livingarea_avg Nullable(Float64),
    nonlivingapartments_avg Nullable(Float64),
    nonlivingarea_avg Nullable(Float64),
    apartments_mode Nullable(Float64),
    basementarea_mode Nullable(Float64),
    years_beginexpluatation_mode Nullable(Float64),
    years_build_mode Nullable(Float64),
    commonarea_mode Nullable(Float64),
    elevators_mode Nullable(Float64),
    entrances_mode Nullable(Float64),
    floorsmax_mode Nullable(Float64),
    floorsmin_mode Nullable(Float64),
    landarea_mode Nullable(Float64),
    livingapartments_mode Nullable(Float64),
    livingarea_mode Nullable(Float64),
    nonlivingapartments_mode Nullable(Float64),
    nonlivingarea_mode Nullable(Float64),
    apartments_medi Nullable(Float64),
    basementarea_medi Nullable(Float64),
    years_beginexpluatation_medi Nullable(Float64),
    years_build_medi Nullable(Float64),
    commonarea_medi Nullable(Float64),
    elevators_medi Nullable(Float64),
    entrances_medi Nullable(Float64),
    floorsmax_medi Nullable(Float64),
    floorsmin_medi Nullable(Float64),
    landarea_medi Nullable(Float64),
    livingapartments_medi Nullable(Float64),
    livingarea_medi Nullable(Float64),
    nonlivingapartments_medi Nullable(Float64),
    nonlivingarea_medi Nullable(Float64),
    fondkapremont_mode Nullable(String),
    housetype_mode Nullable(String),
    totalarea_mode Nullable(Float64),
    wallsmaterial_mode Nullable(String),
    emergencystate_mode Nullable(String),
    obs_30_cnt_social_circle Nullable(Float64),
    def_30_cnt_social_circle Nullable(Float64),
    obs_60_cnt_social_circle Nullable(Float64),
    def_60_cnt_social_circle Nullable(Float64),
    days_last_phone_change Nullable(Float64),
    flag_document_2 Nullable(Int32),
    flag_document_3 Nullable(Int32),
    flag_document_4 Nullable(Int32),
    flag_document_5 Nullable(Int32),
    flag_document_6 Nullable(Int32),
    flag_document_7 Nullable(Int32),
    flag_document_8 Nullable(Int32),
    flag_document_9 Nullable(Int32),
    flag_document_10 Nullable(Int32),
    flag_document_11 Nullable(Int32),
    flag_document_12 Nullable(Int32),
    flag_document_13 Nullable(Int32),
    flag_document_14 Nullable(Int32),
    flag_document_15 Nullable(Int32),
    flag_document_16 Nullable(Int32),
    flag_document_17 Nullable(Int32),
    flag_document_18 Nullable(Int32),
    flag_document_19 Nullable(Int32),
    flag_document_20 Nullable(Int32),
    flag_document_21 Nullable(Int32),
    amt_req_credit_bureau_hour Nullable(Float64),
    amt_req_credit_bureau_day Nullable(Float64),
    amt_req_credit_bureau_week Nullable(Float64),
    amt_req_credit_bureau_mon Nullable(Float64),
    amt_req_credit_bureau_qrt Nullable(Float64),
    amt_req_credit_bureau_year Nullable(Float64)
)
ENGINE = MergeTree
PRIMARY KEY (sk_id_curr);


CREATE TABLE bureau
(
    sk_id_curr Int64,
    sk_id_bureau Int64,
    credit_active Nullable(String),
    credit_currency Nullable(String),
    days_credit Nullable(Int32),
    credit_day_overdue Nullable(Int32),
    days_credit_enddate Nullable(Float64),
    days_enddate_fact Nullable(Float64),
    amt_credit_max_overdue Nullable(Float64),
    cnt_credit_prolong Nullable(Int32),
    amt_credit_sum Nullable(Float64),
    amt_credit_sum_debt Nullable(Float64),
    amt_credit_sum_limit Nullable(Float64),
    amt_credit_sum_overdue Nullable(Float64),
    credit_type Nullable(String),
    days_credit_update Nullable(Int32),
    amt_annuity Nullable(Float64)
)
ENGINE = MergeTree
PRIMARY KEY (sk_id_bureau)
ORDER BY (sk_id_bureau);

CREATE TABLE bureau_balance
(
    sk_id_bureau   Int64,
    months_balance Int32,
    status         Nullable(String),
)
ENGINE = MergeTree
PRIMARY KEY (sk_id_bureau, months_balance)
ORDER BY  (sk_id_bureau, months_balance);

CREATE TABLE previous_application
(
    sk_id_prev                   Int64,
    sk_id_curr                   Int64,
    name_contract_type           String,
    amt_annuity                  Float64,
    amt_application              Float64,
    amt_credit                   Float64,
    amt_down_payment             Float64,
    amt_goods_price              Float64,
    weekday_appr_process_start   String,
    hour_appr_process_start      Int32,
    flag_last_appl_per_contract  String,
    nflag_last_appl_in_day       Int32,
    rate_down_payment            Float64,
    rate_interest_primary        Float64,
    rate_interest_privileged     Float64,
    name_cash_loan_purpose       String,
    name_contract_status         String,
    days_decision                Int32,
    name_payment_type            String,
    code_reject_reason           String,
    name_type_suite              String,
    name_client_type             String,
    name_goods_category          String,
    name_portfolio               String,
    name_product_type            String,
    channel_type                 String,
    sellerplace_area             Int32,
    name_seller_industry         String,
    cnt_payment                  Float64,
    name_yield_group             String,
    product_combination          String,
    days_first_drawing           Float64,
    days_first_due               Float64,
    days_last_due_1st_version    Float64,
    days_last_due                Float64,
    days_termination             Float64,
    nflag_insured_on_approval    Float64
)
ENGINE = MergeTree
PRIMARY KEY (sk_id_prev)
ORDER BY  (sk_id_prev);

CREATE TABLE pos_cash_balance
(
    sk_id_prev            Int64,
    sk_id_curr            Int64,
    months_balance        Int32,
    cnt_instalment        Float64,
    cnt_instalment_future Float64,
    name_contract_status  String,
    sk_dpd                Int32,
    sk_dpd_def            Int32
)
ENGINE = MergeTree
PRIMARY KEY (sk_id_prev)
ORDER BY  (sk_id_prev);

CREATE TABLE credit_card_balance
(
    sk_id_prev                      Int64,
    sk_id_curr                      Int64,
    months_balance                  Int32,
    amt_balance                     Float64,
    amt_credit_limit_actual         Int64,
    amt_drawings_atm_current        Float64,
    amt_drawings_current            Float64,
    amt_drawings_other_current      Float64,
    amt_drawings_pos_current        Float64,
    amt_inst_min_regularity         Float64,
    amt_payment_current             Float64,
    amt_payment_total_current       Float64,
    amt_receivable_principal        Float64,
    amt_recivable                   Float64,
    amt_total_receivable            Float64,
    cnt_drawings_atm_current        Float64,
    cnt_drawings_current            Int32,
    cnt_drawings_other_current      Float64,
    cnt_drawings_pos_current        Float64,
    cnt_instalment_mature_cum       Float64,
    name_contract_status            String,
    sk_dpd                          Int32,
    sk_dpd_def                      Int32
)
ENGINE = MergeTree
PRIMARY KEY (sk_id_prev)
ORDER BY  (sk_id_prev);

CREATE TABLE installments_payments
(
    sk_id_prev              Int64,
    sk_id_curr              Int64,
    num_instalment_version  Float64,
    num_instalment_number   Int32,
    days_instalment         Float64,
    days_entry_payment      Float64,
    amt_instalment          Float64,
    amt_payment             Float64
)
ENGINE = MergeTree
PRIMARY KEY (sk_id_prev)
ORDER BY  (sk_id_prev);
