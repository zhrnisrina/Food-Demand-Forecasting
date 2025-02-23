---
title: "Problem 1"
author: "Putri Nisrina Az-Zahra - M0501241050"
format: 
  html:
    embed-resources: true
    toc: true
    self-contained: true
---

------------------------------------------------------------------------

Demand Forecasting

It is a meal delivery company which operates in multiple cities. They have various fulfillment centers in these cities for dispatching meal orders to their customers. The client wants you to help these centers with demand forecasting for upcoming weeks so that these centers will plan the stock of raw materials accordingly.

Goals : **Predict the demand for the next 10 weeks** (Weeks: 136-145) for the center-meal combinations in the test set ; Forecasting accurately number of order (num_order)

```{r}
# Mengatur working directory
setwd("E:\\S2 IPB\\SEMESTER 2\\Pembelajaran Mesin Statistika\\Praktikum\\Problem 1")
```

## Load Library

```{r}
library(readxl)      # Membaca file Excel 
library(skimr)       # Ringkasan statistik eksploratif
library(ggplot2)     # Visualisasi data berbasis grammar of graphics
library(tidyverse)   # Kumpulan paket untuk manipulasi & analisis data
library(reshape2)    # Transformasi format data (wide ↔ long)
library(corrplot)    # Visualisasi matriks korelasi
library(gridExtra)   # Menggabungkan beberapa plot dalam satu grid
library(caret)       # Pemrosesan data & validasi model
library(data.table)  # Manipulasi data skala besar
library(lightgbm)    # Model Gradient Boosting yang cepat
library(xgboost)     # Model Extreme Gradient Boosting
library(Matrix)      # Operasi matriks efisien
library(randomForest)# Algoritma Random Forest
library(fastDummies) # Pembuatan variabel dummy cepat
```

## Load Data

### Train

```{r}
train <- read_xlsx("train.xlsx")
head(train)
```

### Test

```{r}
test <- read_xlsx("test.xlsx")
head(test)
```

### Meal Info

```{r}
meal_info <- read.csv("meal_info.csv")
head(meal_info)
```

### Fulfilment Center Info

```{r}
fulfilment_center_info <- read.csv("fulfilment_center_info.csv")
head(fulfilment_center_info)
```

## Pre-Processing Data

### Added Meal Info

```{r}
# Join Data Meal Info by meal_id
train <- train %>%
  left_join(meal_info, by = "meal_id")

head(train)
```

### Added Fulfillment Center Info

```{r}
# Join Data Fulfillment Center Info by center_id
train <- train %>%
  left_join(fulfilment_center_info, by = "center_id")

head(train)
```

### Updated Data

```{r}
# Melihat Struktur Data
str(train)
```

Dataset ini berisi 456,548 baris dan 15 kolom yang terdiri atas 11 variabel numerik dan 4 variabel kategorik.

-   `id`: ID unik untuk setiap data.

-   `week`: Periode waktu pemesanan makanan.

-   `center_id`: ID pusat distribusi makanan.

-   `meal_id`: ID makanan yang ditawarkan.

-   `checkout_price`: Harga akhir makanan yang dibayar pelanggan setelah diskon.

-   `base_price`: Harga awal makanan sebelum diskon.

-   `emailer_for_promotion`: Indikator apakah makanan dipromosikan melalui email (0 = tidak, 1 = ya).

-   `homepage_featured`: Indikator apakah makanan ditampilkan di halaman utama untuk promosi (0 = tidak, 1 = ya).

-   `num_orders`: Jumlah pesanan makanan dan berperan sebagai variabel target.

-   `category`: Kategori makanan atau minuman.

-   `cuisine`: Jenis masakan berdasarkan gaya kuliner.

-   `city_code`: Kode kota tempat pusat distribusi berada.

-   `region_code`: Kode wilayah tempat pusat distribusi berada.

-   `center_type`: Jenis pusat distribusi makanan.

-   `op_area`: Luas operasional pusat distribusi dalam satuan tertentu.

```{r}
# Mengecek Missing Value pada data train
any(is.na(train))
```

Tidak ada variabel yang memiliki missing value (NA) pada data train sehingga tidak perlu dilakukan penanganan seperti imputasi atau penghapusan baris/kolom dengan nilai NA.

```{r}
# Split Data as Validation
test_validation <- train %>% filter((week >= 136 & week <= 145))
glimpse(test_validation)
```

Data train dengan minggu ke-136 sampai minggu ke-145 digunakan untuk validasi jumlah num_orders dari hasil prediksi model yaitu sebanyak 32.821 baris data.

```{r}
# Data Week 1 - 135
train <- train %>% filter(!(week >= 136 & week <= 145))
glimpse(train)
```

Data train dengan minggu ke-1 sampai minggu ke-135 sebanyak 423.727 baris dan 15 kolom digunakan untuk melatih model yang dapat memprediksi variabel num_orders atau jumlah pesanan makanan.

## Exploration Data

### Summary Data

```{r}
skim_without_charts(train)
```

### Check Duplicated Data

```{r}
# Mengecek duplikasi data
sum(duplicated(train))
```

Tidak ada data yang duplikat pada data train sehingga tidak perlu dilakukan penghapusan baris data yang duplikat.

### Num of Order

```{r}
train_filtered <- train %>%
  group_by(week) %>%
  summarize(avg_orders = mean(num_orders, na.rm = TRUE))

ggplot(train_filtered, aes(x = week, y = avg_orders)) +
  geom_line(color = "#48A6A7", linewidth = 1) + 
  labs(title = "Average Number of Orders", x = "Week", y = "Avg Orders") +
  theme_classic() + 
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold")
  )
```

Plot deret waktu dari `num_orders` menunjukkan **fluktuasi yang signifikan** dalam jumlah pesanan makanan per minggu, dengan beberapa lonjakan tajam yang kemungkinan dipengaruhi oleh promosi atau faktor musiman. Sebaliknya, terdapat pula penurunan drastis pada beberapa titik. Pola ini menunjukkan kemungkinan **adanya siklus permintaan yang berulang**, meskipun di bagian akhir grafik tren tampak lebih stabil.

```{r}
ggplot(train, aes(x = num_orders)) +
  geom_histogram(binwidth = 10, fill = "#48A6A7", color = "#48A6A7", alpha = 0.8) + 
  labs(title = "Distribution of Number of Orders", x = "Number of Orders", y = "Count") +
  theme_classic() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold")
  )
```

Plot ini menunjukkan distribusi jumlah pesanan makanan dengan sebagian besar pesanan berada pada nilai yang relatif kecil. Sebagian besar **jumlah pesanan terkonsentrasi di sekitar nol** hingga beberapa ratus pesanan, sementara ada beberapa outlier dengan jumlah pesanan yang jauh lebih tinggi, bahkan mendekati 25.000. Distribusi ini mengindikasikan bahwa sebagian besar makanan memiliki permintaan yang rendah hingga sedang, sementara hanya sedikit yang memiliki permintaan sangat tinggi.

## Feature

### Density Plot of Numeric Feature

```{r}
train %>%
  select(-num_orders) %>%
  select(where(is.numeric)) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  facet_wrap(~ variable, scales = "free") +
  geom_density(fill = "#48A6A7", alpha = 0.5) +
  labs(
    title = "Density Plot of Numeric Feature",
    x = "Value",
    y = "Density"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold")
  )
```

-   `base_price` dan `checkout_price` : menunjukkan distribusi multimodal, menunjukkan adanya beberapa kelompok harga yang dominan.

-   `center_id` dan `city_code` : menunjukkan pola multimodal, mengindikasikan adanya beberapa pusat distribusi dan kode kota dengan jumlah yang cukup banyak.

-   `emailer_for_promotion` dan `homepage_featured` : memiliki distribusi yang sangat miring ke kiri, menunjukkan bahwa sebagian besar nilai adalah nol, menandakan hanya sedikit item yang dipromosikan melalui email atau ditampilkan di halaman utama.

-   `id` dan `meal_id` : memiliki distribusi yang hampir uniform, mengindikasikan bahwa nilai-nilainya tersebar merata dalam rentang tertentu.

-   `op_area` dan `region_code` : menunjukkan distribusi multimodal, mengindikasikan adanya beberapa kategori operasional dan wilayah dengan karakteristik berbeda.

-   `week` : memiliki distribusi yang cukup merata, yang mungkin menunjukkan bahwa data mencakup periode waktu yang cukup panjang tanpa banyak variasi dalam cakupan minggu.

### Boxplot of Numeric Feature

```{r}
train %>%
  select(-num_orders) %>% 
  select(where(is.numeric)) %>%  
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = "", y = value)) +
  geom_boxplot(fill = "#48A6A7", color = "black") +
  theme_minimal() +
  labs(title = "Boxplot of Numeric Feature",
       x = "",
       y = "Value") +
  facet_wrap(~ variable, scales = "free", ncol = 4) + 
  theme(
    axis.text.x = element_blank(), 
    axis.ticks.x = element_blank(),  
    strip.text = element_text(face = "bold"),  
    plot.title = element_text(hjust = 0.5, face = "bold") 
  )
```

-   `base_price` dan `checkout_price` : memiliki distribusi dengan beberapa pencilan (outlier) di bagian atas, menunjukkan adanya beberapa harga yang jauh lebih tinggi dibandingkan mayoritas data.

-   `emailer_for_promotion` dan `homepage_featured` : memiliki distribusi yang sangat tidak merata dengan dominasi nilai nol, tetapi ada beberapa pencilan yang menunjukkan bahwa hanya sebagian kecil data yang memiliki nilai 1.

-   `op_area` : menunjukkan beberapa outlier, baik di bagian bawah maupun atas, menandakan variasi yang cukup besar dalam area operasional.

-   `id`, `meal_id`, `region_code`, dan `week` : memiliki distribusi yang lebih merata tanpa outlier signifikan, menunjukkan bahwa data pada fitur ini tersebar secara konsisten dalam rentangnya.

-   `center_id` dan `city_code` : menunjukkan distribusi yang cukup lebar tetapi tidak memiliki pencilan signifikan, menandakan bahwa pusat distribusi dan kode kota tersebar dalam kisaran yang cukup stabil.

### Correlation between num_orders and Numeric Features

```{r}
correlations <- train %>%
  select(num_orders, where(is.numeric)) %>%
  cor(use = "complete.obs")

correlations_num_orders <- correlations["num_orders", -1]  

correlations_table <- data.frame(
  Variable = names(correlations_num_orders),
  Correlation = round(as.numeric(correlations_num_orders), 3) # Pembulatan 3 desimal
)

print(correlations_table, row.names = FALSE)

```

Korelasi antara variabel-variabel ini dengan `num_order` relatif rendah. **Tidak ada variabel yang memiliki korelasi yang sangat kuat** (\> 0.5 atau \< -0.5), yang menunjukkan bahwa tidak ada satu faktor dominan yang secara langsung menentukan jumlah pesanan.

### Correlation Between Numeric Feature

```{r}
numeric_vars <- 
  train %>% 
  select(-num_orders) %>%
  select(where(is.numeric))
correlation_matrix_pearson <- cor(numeric_vars, method = "pearson")
corrplot(correlation_matrix_pearson, method = "color", type = "lower",
         col = colorRampPalette(c("darkgreen", "white", "darkorange"))(200),
         tl.col = "black", tl.cex = 0.5, 
         addCoef.col = "black", number.cex = 0.6, number.font = 0.5)

```

Plot heatmap menunjukkan bahwa s**ebagian besar variabel tidak memiliki korelasi kuat satu sama lain**. Variabel harga (`checkout_price` dan `base_price`) memiliki korelasi tinggi, yang menunjukkan **keterkaitan langsung antara harga dasar dan harga jual.**

### Distribution of Category Feature

```{r}
category_counts <- train %>%
  count(category) %>%
  arrange(desc(n))  

ggplot(category_counts, aes(x = reorder(category, -n), y = n, fill = n)) +
  geom_bar(stat = "identity") + 
  theme_minimal() +
  labs(title = "Distribution of Category",
       x = "Category",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank(), 
        panel.background = element_blank(),
        plot.title = element_text(hjust = 0.5, face = "bold")) +
  scale_fill_gradient(low = "#88A9A1", high = "#48A6A7")
```

Grafik ini menunjukkan distribusi jumlah pesanan berdasarkan kategori makanan dan minuman. Terlihat bahwa **kategori Beverages memiliki jumlah pesanan yang jauh lebih tinggi** dibandingkan kategori lainnya, dengan lebih dari 100.000 pesanan, menjadikannya produk paling populer dalam dataset ini.

```{r}
category_counts <- train %>%
  count(cuisine) %>%
  mutate(percentage = n / sum(n) * 100)

p1 <- ggplot(category_counts, aes(x = "", y = n, fill = cuisine)) +
  geom_bar(stat = "identity", width = 1, color = "black") +  # Added border color
  coord_polar("y") + 
  theme_void() + 
  labs(title = "Pie Chart of Cuisine") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
         legend.position = "bottom") +
  scale_fill_brewer(palette = "GnBu") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), position = position_stack(vjust = 0.5), color = "black")

category_counts <- train %>%
  count(center_type) %>%
  mutate(percentage = n / sum(n) * 100)

p2 <- ggplot(category_counts, aes(x = "", y = n, fill = center_type)) +
  geom_bar(stat = "identity", width = 1, color = "black") +  # Added border color
  coord_polar("y") + 
  theme_void() + 
  labs(title = "Pie Chart of Center Type") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
         legend.position = "bottom") +
  scale_fill_brewer(palette = "BuGn") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), position = position_stack(vjust = 0.5), color = "black")


grid.arrange(p1, p2, ncol = 2)

```

Pada pie chart dengan variabel cuisine, terlihat bahwa **masakan Italian memiliki proporsi terbesar**, yaitu 26.9%, diikuti oleh masakan Thai sebesar 25.9%. Sementara itu, masakan Indian mencakup 24.7% dari total, dan masakan Continental memiliki proporsi terkecil dengan 22.5%.

Pada pie chart dengan variabel center_type, distribusi pusat distribusi berdasarkan jenisnya menunjukkan bahwa TYPE_A mendominasi dengan 57.6% dari total pusat distribusi, sedangkan TYPE_B dan TYPE_C masing-masing memiliki proporsi 20.6% dan 21.8%. Hal ini menunjukkan bahwa **sebagian besar operasi distribusi dikelola oleh pusat dengan tipe A**, yang kemungkinan memiliki kapasitas lebih besar atau cakupan yang lebih luas dibandingkan tipe lainnya.

```{r}
category_orders <- train %>%
  group_by(category) %>%
  summarise(total_orders = mean(num_orders, na.rm = TRUE)) %>%
  arrange(desc(total_orders))

ggplot(category_orders, aes(x = reorder(category, -total_orders), y = total_orders, fill = total_orders)) +
  geom_bar(stat = "identity") + 
  theme_minimal() +
  labs(title = "Mean Orders per Category",
       x = "Category",
       y = "Mean Num Orders") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank(), 
        panel.background = element_blank(),
        plot.title = element_text(hjust = 0.5, face = "bold")) +
  scale_fill_gradient(low = "#88A9A1", high = "#48A6A7")
```

Pada bar plot dengan variabel `category`, kategori **Rice Bowl memiliki rata-rata pesanan tertinggi**, diikuti oleh Sandwich dan Salad, yang juga memiliki jumlah pesanan yang cukup tinggi dibandingkan kategori lainnya. Beverages dan Extras juga memiliki jumlah pesanan yang signifikan, meskipun lebih rendah dibandingkan tiga kategori teratas.

```{r}
p1 <- ggplot(train, aes(x = center_type, y = num_orders, fill = center_type)) +
  geom_boxplot(alpha = 0.7, outlier.color = "black", outlier.shape = 16) + 
  theme_minimal() +
  labs(title = "Orders by Center Type",
       x = "",
       y = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank(), 
        panel.background = element_blank(),
        legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold")) +
  scale_fill_brewer(palette = "Set3")

p2 <- ggplot(train, aes(x = cuisine, y = num_orders, fill = cuisine)) +
  geom_boxplot(alpha = 0.7, outlier.color = "black", outlier.shape = 16) + 
  theme_minimal() +
  labs(title = "Orders by Cuisine",
       x = "",
       y = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank(), 
        panel.background = element_blank(),
        legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold")) +
  scale_fill_brewer(palette = "Set3")

grid.arrange(p1, p2, ncol = 2)
```

Pada Boxplot Orders by Center Type, terdapat tiga kategori pusat distribusi, yaitu TYPE_A, TYPE_B, dan TYPE_C. M**asing-masing kategori memiliki persebaran jumlah pesanan yang cukup mirip**, dengan banyak outlier yang menunjukkan bahwa beberapa pusat distribusi memiliki pesanan yang jauh lebih tinggi dibandingkan mayoritas lainnya.

Pada Boxplot Orders by Cuisine, terdapat empat kategori masakan yaitu Continental, Indian, Italian, dan Thai. **Pola persebarannya juga menunjukkan bahwa sebagian besar pesanan berada di kisaran rendah**, namun terdapat beberapa outlier dengan jumlah pesanan yang sangat tinggi, terutama pada masakan Indian dan Italian.

```{r}
center_orders <- train %>%
  group_by(center_type) %>%
  summarise(total_orders = mean(num_orders, na.rm = TRUE)) %>%
  arrange(desc(total_orders))

p1 <- ggplot(center_orders, aes(x = reorder(center_type, -total_orders), y = total_orders, fill = center_type)) +
  geom_bar(stat = "identity", alpha = 0.7) + 
  theme_minimal() +
  labs(title = "Mean Orders by Center Type",
       x = "",
       y = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank(), 
        panel.background = element_blank(),
        legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold")) +
  scale_fill_brewer(palette = "Set3")

cuisine_data <- train %>%
  group_by(cuisine) %>%
  summarise(total_orders = mean(num_orders, na.rm = TRUE)) %>%
  arrange(desc(total_orders))

p2 <- ggplot(cuisine_data, aes(x = reorder(cuisine, -total_orders), y = total_orders, fill = cuisine)) +
  geom_bar(stat = "identity", alpha = 0.7) + 
  theme_minimal() +
  labs(title = "Mean Orders by Cuisine",
       x = "",
       y = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank(), 
        panel.background = element_blank(),
        legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold")) +
  scale_fill_brewer(palette = "Set3")

grid.arrange(p1, p2, ncol = 2)

```

Pada barplot Mean Orders by Center Type, menunjukkan bahwa **pusat distribusi TYPE_B memiliki rata-rata jumlah pesanan tertinggi**, diikuti oleh TYPE_A, sementara TYPE_C memiliki rata-rata terendah. Hal ini dapat mengindikasikan bahwa pusat distribusi TYPE_B lebih dominan dalam pemenuhan pesanan dibandingkan dua tipe lainnya.

Pada barplot Mean Orders by Cuisine mengilustrasikan bahwa **masakan Italian memiliki rata-rata pesanan tertinggi**, diikuti oleh Thai dan Indian, sedangkan Continental memiliki rata-rata pesanan terendah. Ini menunjukkan bahwa masakan Italian lebih populer atau memiliki permintaan yang lebih tinggi dibandingkan jenis masakan lainnya.

## Pre-Modelling

### Split Train - Test Data

Dataset train menjadi **data latih (80%) dan data uji (20%)** secara stratifikasi berdasarkan `id`. Fungsi set.seed(50) memastikan pemilihan data tetap konsisten, sementara createDataPartition() menghasilkan indeks untuk data latih. Indeks ini kemudian digunakan untuk mengekstrak data latih dan data uji dari dataset asli.

```{r}
set.seed(50)
train_indices <- createDataPartition(train$id, p = 0.8, list = FALSE)
train_data <- train[train_indices, ]
test_data <- train[-train_indices, ]
```

### Declare Target and Feature

Selanjutnya dilakukan pemisahan fitur dan target dalam dataset pelatihan dan pengujian. Setelah itu, data dipisahkan menjadi variabel yang berisi fitur dan variabel yang berisi target sehingga siap digunakan untuk proses pemodelan.

```{r}
features <- setdiff(names(train), c("id", "num_orders"))
X_train <- train_data[, features, with = FALSE]
y_train <- train_data$num_orders
X_test <- test_data[, features, with = FALSE]
y_test <- test_data$num_orders
```

### One-Hot Encoding

Data diubah menjadi format data.table, kemudian fitur kategorikal diidentifikasi dan dikonversi menjadi faktor. Selanjutnya, dilakukan **One-Hot Encoding untuk mengonversi kategori menjadi variabel biner**. Setelah itu, semua kolom dipastikan dalam bentuk numerik untuk melakukan pemodelan machine learning. Terakhir, dilakukan pengecekan untuk memastikan tidak ada nilai yang hilang (NA) dalam data yang telah diproses.

```{r}
# Memastikan X_train dan X_test adalah data.table
setDT(X_train)
setDT(X_test)

# Identifikasi fitur kategorikal yang valid
categorical_features <- names(X_train)[sapply(X_train, is.character)]
categorical_features <- intersect(names(X_train), categorical_features)

# Konversi fitur kategorikal menjadi faktor
X_train[, (categorical_features) := lapply(.SD, as.factor), .SDcols = categorical_features]
X_test[, (categorical_features) := lapply(.SD, as.factor), .SDcols = categorical_features]

# Pastikan categorical_features hanya berisi fitur yang ada di X_train_final
setdiff(categorical_features, names(X_train)) 
```

```{r}
# One-Hot Encoding untuk fitur kategorikal pada X_train dan X_test
X_train_encoded <- dummy_cols(X_train, select_columns = categorical_features, remove_first_dummy = TRUE)
X_test_encoded <- dummy_cols(X_test, select_columns = categorical_features, remove_first_dummy = TRUE)

# Pisahkan kembali target (num_orders)
X_train_final <- X_train_encoded[, setdiff(names(X_train_encoded), "num_orders"), with = FALSE]
X_test_final <- X_test_encoded[, setdiff(names(X_test_encoded), "num_orders"), with = FALSE]

# Pastikan semua kolom dalam data adalah numerik setelah encoding
X_train_final <- data.matrix(X_train_final)  
X_test_final <- data.matrix(X_test_final)

# Konfirmasi bahwa tidak ada NA di data
sum(is.na(X_train_final))  
sum(is.na(X_test_final))   
```

## Modelling

### 1. LightGBM

```{r}
# Membuat objek dataset LightGBM
dtrain <- lgb.Dataset(
  data = as.matrix(X_train_final),  
  label = y_train
)

dtest <- lgb.Dataset(
  data = as.matrix(X_test_final),  
  label = y_test,
  reference = dtrain
)

# Parameter LightGBM
params <- list(
  objective = "regression",
  metric = "rmse",
  boosting = "gbdt",
  learning_rate = 0.05, 
  num_leaves = 64,  
  max_depth = -1,  
  feature_fraction = 0.7,  
  bagging_fraction = 0.7,  
  bagging_freq = 5
)


# Cross Validation
cv_results <- lgb.cv(
  params = params,
  data = dtrain,
  nfold = 5,
  nrounds = 100,
  early_stopping_rounds = 10,
  verbose = 1
)

# Melatih model akhir dengan iterasi terbaik dari CV
model_lgb <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = cv_results$best_iter
)

# Model telah dilatih
model_lgb
```

Model LightGBM telah dilatih dengan **100 pohon dan 31 fitur** tanpa memenuhi kondisi early stopping, yang berarti model berjalan hingga jumlah maksimum iterasi yang ditentukan. Model ini Tidak adanya early stopping menunjukkan bahwa jumlah pohon yang digunakan mungkin masih dapat dioptimalkan lebih lanjut.

```{r}
# Melakukan prediksi pada data uji
preds_lgb <- predict(model_lgb, as.matrix(X_test_final))  

# Menghitung RMSE
rmse <- sqrt(mean((preds_lgb - y_test)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

# Menghitung R-squared (R²)
sst <- sum((y_test - mean(y_test))^2)  
sse <- sum((y_test - preds_lgb)^2)       
r2_lgb <- 1 - (sse / sst)                  
cat("R-squared (R²):", r2_lgb, "\n")

```

Nilai R-squared (R²) sebesar `r round(r2_lgb, 4)` menunjukkan bahwa model LightGBM mampu menjelaskan sekitar **`r round(r2_lgb*100, 2)`% variasi dalam data target**, sementara sisanya, sekitar `r round(100 - r2_lgb*100, 2)`%, tidak terjelaskan oleh model. Hal ini mengindikasikan bahwa model memiliki performa yang cukup baik, meskipun masih dapat dilakukan perbaikan.

### 2. XGBoost

```{r}
# Membuat objek DMatrix untuk XGBoost
dtrain <- xgb.DMatrix(
  data = as.matrix(X_train_final),  # Gunakan data yang sudah di-encode
  label = y_train
)

dtest <- xgb.DMatrix(
  data = as.matrix(X_test_final),  # Gunakan data yang sudah di-encode
  label = y_test
)

# Parameter XGBoost
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  booster = "gbtree",
  eta = 0.05,
  max_depth = 8,
  min_child_weight = 10,
  gamma = 0.1,
  lambda = 1,
  alpha = 0.5,
  subsample = 0.7,
  colsample_bytree = 0.7,
  nthread = 4,
  verbosity = 1
)

# Cross Validation
cv_results <- xgb.cv(
  params = params,
  data = dtrain,
  nfold = 5,
  nrounds = 100,
  early_stopping_rounds = 10,
  verbose = 1
)

# Melatih model akhir dengan iterasi terbaik dari CV
model_xgb <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = cv_results$best_iteration
)

# Model telah dilatih
model_xgb

```

Model XGBoost telah dilatih menggunakan **100 iterasi dengan 31 fitur** menggunakan parameter yang telah ditentukan. Model ini menggunakan metrik evaluasi rmse untuk regresi.

```{r}
# Melakukan prediksi pada data uji
preds_xgb <- predict(model_xgb, as.matrix(X_test_final))  

# Menghitung RMSE
rmse <- sqrt(mean((preds_xgb - y_test)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

# Menghitung R-squared (R²)
sst <- sum((y_test - mean(y_test))^2)  
sse <- sum((y_test - preds_xgb)^2)       
r2_xgb <- 1 - (sse / sst)                  
cat("R-squared (R²):", r2_xgb, "\n")

```

Model XGBoost menghasilkan nilai R-squared (R²) sebesar `r round(r2_xgb, 4)` yang menunjukkan bahwa model dapat menjelaskan sekitar **`r round(r2_xgb*100, 2)`% variasi dalam data target**, sementara sisanya sekitar `r round(100-r2_xgb*100, 2)`% tidak dapat dijelaskan oleh model.

### 3. Random Forest

```{r}
# Membuat objek data frame untuk Random Forest
dtrain <- data.frame(X_train_final, y_train)
dtest <- data.frame(X_test_final, y_test)

# Melatih model Random Forest
model_rf <- randomForest(
  x = X_train_final,  
  y = y_train,
  ntree = 400,
  mtry = sqrt(ncol(X_train_final)),  
  importance = TRUE
)

# Model telah dilatih
model_rf

```

```{r}
# Melakukan prediksi pada data uji
preds_rf <- predict(model_rf, X_test_final)  

# Menghitung RMSE
rmse <- sqrt(mean((preds_rf - y_test)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

# Menghitung R-squared (R²)
sst <- sum((y_test - mean(y_test))^2)  
sse <- sum((y_test - preds_rf)^2)       
r2_rf <- 1 - (sse / sst)                  
cat("R-squared (R²):", r2_rf, "\n")

```

Model Random Forest menunjukkan kinerja yang cukup baik dengan R² sebesar `r round(r2_rf, 4)`, yang berarti model dapat menjelaskan **`r round(r2_rf*100, 2)`% variasi dalam data target**. Namun, RMSE sebesar `r round(rmse, 2)` menunjukkan bahwa rata-rata kesalahan prediksi masih cukup besar, yang mengindikasikan peluang untuk peningkatan model.

### 4. Ensemble - Averaging

```{r}
# Menormalisasi bobot agar totalnya menjadi 1
total_r2 <- r2_lgb + r2_xgb + r2_rf
w_lgb <- r2_lgb / total_r2
w_xgb <- r2_xgb / total_r2
w_rf <- r2_rf / total_r2

# Weighted Ensemble Prediction
pred_ensemble <- (w_lgb * preds_lgb) + (w_xgb * preds_xgb) + (w_rf * preds_rf)

# Menghitung RMSE
rmse_ens <- sqrt(mean((pred_ensemble - y_test)^2))
cat("Root Mean Squared Error (RMSE):", rmse_ens, "\n")

# Menghitung R-squared (R²)
sst <- sum((y_test - mean(y_test))^2)  
sse <- sum((y_test - pred_ensemble)^2)       
r2_ens <- 1 - (sse / sst)                  
cat("R-squared (R²):", r2_ens, "\n")
```

Hasil ensemble averaging menunjukkan peningkatan performa dibandingkan model individual. Dengan RMSE sebesar `r round(rmse, 2)`, model ensemble memiliki kesalahan rata-rata yang lebih kecil dibandingkan model sebelumya, yang menunjukkan peningkatan akurasi prediksi. Selain itu, R² sebesar `r round(r2_ens, 4)` menunjukkan bahwa model ini mampu menjelaskan **`r round(r2_xgb*100, 2)`% variasi dalam data target. Peningkatan ini membuktikan bahwa menggabungkan LightGBM, XGBoost, dan Random Forest** dengan evaluasi berbasis R² dapat menghasilkan prediksi yang lebih stabil dan akurat.

### 5. Stacking

```{r}
# Gabungkan prediksi sebagai fitur baru
stack_train <- data.frame(
  y_train,
  preds_lgb = predict(model_lgb, as.matrix(X_train_final)),
  preds_xgb = predict(model_xgb, as.matrix(X_train_final)),
  preds_rf  = predict(model_rf, X_train_final)
)

stack_test <- data.frame(
  y_test,
  preds_lgb = predict(model_lgb, as.matrix(X_test_final)),
  preds_xgb = predict(model_xgb, as.matrix(X_test_final)),
  preds_rf  = predict(model_rf, X_test_final)
)

# Meta-model (Linear Regression atau Ridge)
meta_model <- train(y_train ~ ., data = stack_train, 
                    method = "gbm", 
                    trControl = trainControl(method = "cv", number = 5))
```

```{r}
# Prediksi menggunakan meta-model
preds_stack <- predict(meta_model, stack_test)

# Menghitung RMSE
rmse <- sqrt(mean((preds_stack - y_test)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

# Menghitung R-squared (R²)
sst <- sum((y_test - mean(y_test))^2)  
sse <- sum((y_test - preds_stack)^2)       
r2_stack <- 1 - (sse / sst)                  
cat("R-squared (R²):", r2_stack, "\n")
```

Metode stacking dengan Gradient Boosting Machine (GBM) sebagai meta-model menunjukkan peningkatan performa yang signifikan dibandingkan pendekatan individual maupun ensemble averaging. Dengan RMSE sebesar `r round(rmse, 2)`, model ini memiliki kesalahan rata-rata yang lebih kecil dibandingkan ensemble averaging, menunjukkan peningkatan akurasi prediksi. Selain itu, R² sebesar `r round(r2_stack, 4)` menunjukkan bahwa model mampu menjelaskan **`r round(r2_stack*100, 2)`% variasi dalam data target. Peningkatan ini mengonfirmasi bahwa stacking dengan meta-model yang kuat dapat lebih efektif dalam menggabungkan kekuatan model dasar, menghasilkan prediksi yang lebih akurat dan stabil.**

### Comparing R²

```{r}
# Membuat Data Frame untuk R-squared
r2_results <- data.frame(
  Model = c("LightGBM", "XGBoost", "Random Forest", "Averanging", "Stacking"),
  R_squared = c(r2_lgb, r2_xgb, r2_rf,r2_ens, r2_stack) 
)

# Menampilkan tabel
print(r2_results)
```

Berdasarkan hasil pemodelan, metode stacking menunjukkan performa terbaik dengan **R² sebesar `r round(r2_stack*100, 2)`%**, lebih tinggi dibandingkan model individu maupun ensemble- averaging. Maka, **model stacking akan digunakan untuk melakukan prediksi num_orders pada week 135-145** karena mampu memberikan hasil yang lebih akurat dan stabil dalam menjelaskan variabilitas data.

## Feature Importance

### 1. LightGBM

```{r}
# Import feature importance
importance_lgb <- lgb.importance(model_lgb, percentage = TRUE)

# Konversi ke dataframe untuk plotting manual
importance_lgb$Feature <- factor(importance_lgb$Feature, levels = rev(importance_lgb$Feature))

# Plot dengan ggplot2 dan warna kustom
ggplot(importance_lgb[1:10, ], aes(x = Gain, y = Feature)) +
  geom_col(fill = "#48A6A7") +
  labs(title = "Feature Importance LightGBM", x = "Gain", y = "Feature") +
  theme_minimal()

```

Berdasarkan grafik feature importance model LightGBM, fitur yang paling berpengaruh dalam memprediksi jumlah pesanan (`num_orders`) adalah `checkout_price`, diikuti oleh `op_area` dan kategori makanan seperti `category_Rice_Bowl`. Hal ini menunjukkan bahwa **harga akhir dari suatu produk memiliki pengaruh terbesar terhadap jumlah pesanan** karena harga sering kali menjadi faktor utama dalam keputusan pembelian. Selain itu, `op_area` berada di urutan kedua yang mengindikasikan bahwa **lokasi atau cakupan layanan memengaruhi jumlah pesanan**. Fitur seperti `homepage_featured` dan `emailer_for_promotion` juga memiliki peran penting, yang mencerminkan pengaruh promosi dan visibilitas di halaman utama terhadap permintaan. Fitur lainnya seperti `base_price`, `meal_id`, dan `center_id` berkontribusi lebih rendah, namun tetap relevan dalam membangun model. Hal ini dapat menunjukkan pentingnya mengoptimalkan harga, strategi promosi, dan distribusi geografis untuk meningkatkan jumlah pesanan.

### 2. XGBoost

```{r}
# Mendapatkan feature importance
importance_xgb <- xgb.importance(feature_names = colnames(X_train_final), model = model_xgb)

# Konversi ke dataframe untuk plotting manual
importance_xgb$Feature <- factor(importance_xgb$Feature, levels = rev(importance_xgb$Feature))

# Plot dengan ggplot2 dan warna kustom
ggplot(importance_xgb[1:10, ], aes(x = Gain, y = Feature)) +
  geom_col(fill = "#48A6A7") +
  labs(title = "Feature Importance XGBoost", x = "Gain", y = "Feature") +
  theme_minimal()


```

Berdasarkan grafik feature importance dari model XGBoost, fitur yang memiliki pengaruh paling besar dalam memprediksi jumlah pesanan (`num_orders`) adalah `checkout_price`, diikuti oleh `op_area` dan kategori makanan seperti `category_Rice_Bowl`. **Kondisi ini mirip dengan model LightGBM sebelumnya.** Fitur `op_area` juga berada di urutan kedua, yang mengindikasikan bahwa lokasi operasional atau cakupan layanan memiliki pengaruh signifikan terhadap permintaan. Fitur lainnya seperti `homepage_featured` dan `emailer_for_promotion` juga memiliki kontribusi yang penting, mencerminkan bahwa promosi dan visibilitas di halaman utama berdampak pada peningkatan jumlah pesanan. Selain itu, fitur seperti `base_price`, `category_Sandwich`, dan `cuisine_Italian` relevan untuk memahami pola permintaan. Kondisi ini dapat menunjukkan mengoptimalkan harga, lokasi layanan, dan strategi promosi untuk meningkatkan jumlah pesanan.

### 3. Random Forest

```{r}
### Feature Importance untuk Random Forest
importance_rf <- importance(model_rf)
varImpPlot(model_rf, n.var = 10, main = "Feature Importance Random Forest")

```

Berdasarkan grafik feature importance dari model Random Forest, ada dua metrik utama yang digunakan, yaitu %IncMSE (**peningkatan Mean Squared Error jika fitur diacak**) dan IncNodePurity (**peningkatan purity node dalam pohon keputusan**).

Berdasarkan %IncMSE, Fitur `op_area` memiliki pengaruh terbesar terhadap akurasi model, yang menunjukkan bahwa **area operasi sangat penting dalam memprediksi jumlah pesanan** (`num_orders`). Fitur lain yang signifikan adalah `center_id`, `city_code`, dan `homepage_featured`, yang mengindikasikan **bahwa faktor lokasi, pusat distribusi, dan visibilitas di halaman utama memengaruhi hasil prediksi.** `checkout_price` juga memiliki pengaruh besar, yang sesuai dengan pentingnya harga dalam keputusan pembelian.

Dari segi node purity, fitur `checkout_price` menunjukkan kontribusi terbesar, diikuti oleh `homepage_featured`, `base_price`, dan `op_area.` Hal ini menunjukkan bahwa **fitur-fitur tersebut secara signifikan membantu membagi data menjadi kelompok-kelompok yang lebih homogen** di sepanjang pohon keputusan.

Secara keseluruhan, `checkout_price` dan `op_area` konsisten sebagai fitur yang sangat penting dalam kedua metrik ini. Hal ini memberikan petunjuk bahwa **strategi harga, cakupan operasional, serta promosi memiliki dampak besar terhadap jumlah pesanan.**

## Prediction

### Pre-Processing Data

Data Testing dilakukan pre-processing data seperti yang sebelumnya sudah dilakukan pada data training untuk membangkit model. Hal ini dilakukan untuk **memastikan konsistensi antara data training dan testing** sehingga model yang telah dilatih pada data training dapat memproses data testing dengan karakteristik yang serupa.

```{r}
test <- test %>%
  left_join(meal_info, by = "meal_id")

head(test)
```

```{r}
test <- test %>%
  left_join(fulfilment_center_info, by = "center_id")

head(test)
```

```{r}
str(test)
```

```{r}
# Memastikan X_train dan X_test adalah data.table
setDT(test)

# Identifikasi fitur kategorikal yang valid
categorical_features <- names(test)[sapply(test, is.character)]
categorical_features <- intersect(names(test), categorical_features)

# Konversi fitur kategorikal menjadi faktor
test[, (categorical_features) := lapply(.SD, as.factor), .SDcols = categorical_features]

# Pastikan categorical_features hanya berisi fitur yang ada di X_train_final
setdiff(categorical_features, names(test))  # Harusnya output kosong
```

```{r}
# One-Hot Encoding untuk fitur kategorikal pada X_train dan X_test
test_encoded <- dummy_cols(test, select_columns = categorical_features, remove_first_dummy = TRUE)

# Pastikan semua kolom dalam data adalah numerik setelah encoding
test_final <-test_encoded %>% 
  select(-id)

test_final <- data.matrix(test_final)
head(test_final)
```

```{r}
# Konfirmasi bahwa tidak ada NA di data
sum(is.na(test_final))  # Harusnya 0
```

Data pada minggu ke-136 sampai minggu ke-145 yang sudah ada dalam data training sebelumnya akan digunakan sebagai validasi dari hasil prediksi `num_orders`.

```{r}
y_test <- test_validation$num_orders
```

### Stacking

Metode stacking dengan **Gradient Boosting Machine (GBM) sebagai meta-model** akan dilakukan untuk prediksi karena memiliki performa terbaik dibandingkan dengan model-model lainnya.

```{r}
stack_test <- data.frame(
  y_test,
  preds_lgb = predict(model_lgb, as.matrix(test_final)),
  preds_xgb = predict(model_xgb, as.matrix(test_final)),
  preds_rf  = predict(model_rf, test_final)
)
```

```{r}
# Prediksi menggunakan meta-model
preds_stack <- predict(meta_model, stack_test)

# Menghitung RMSE
rmse <- sqrt(mean((preds_stack - y_test)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

# Menghitung R-squared (R²)
sst <- sum((y_test - mean(y_test))^2)  
sse <- sum((y_test - preds_stack)^2)       
r2_stack <- 1 - (sse / sst)                  
cat("R-squared (R²):", r2_stack, "\n")
```

Hasil prediksi menggunakan metode stacking dengan Gradient Boosting Machine (GBM) sebagai meta-model menunjukkan performa yang cukup baik. Nilai Root Mean Squared Error (RMSE) sebesar `r round(rmse, 2)` menunjukkan bahwa rata-rata kesalahan prediksi model terhadap nilai aktual berada dalam kisaran tersebut. Sementara itu, nilai **R-squared (R²) sebesar `r round(r2_stack, 4)` mengindikasikan bahwa model dapat menjelaskan sekitar `r round(r2_stack*100, 2)`% variasi dalam validasi hasil prediksi.** Ini menunjukkan bahwa pendekatan stacking berhasil menggabungkan keunggulan dari model LightGBM, XGBoost, dan Random Forest untuk meningkatkan akurasi prediksi dibandingkan dengan model individu. Namun, masih terdapat sekitar `r round(100-r2_stack*100, 2)`% variasi yang tidak dapat dijelaskan oleh model, yang bisa menjadi **peluang untuk perbaikan lebih lanjut, misalnya dengan melakukan tuning hyperparameter atau menambahkan fitur tambahan.**


