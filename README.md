# Лабораторная работа №2. Реализация глубокой нейронной сети
**Данные**: В работе предлагается использовать набор данных notMNIST, который состоит из изображений размерностью 28×28 первых 10 букв латинского алфавита (A … J, соответственно). Обучающая выборка содержит порядка 500 тыс. изображений, а тестовая – около 19 тыс.

Данные можно скачать по ссылке:

- https://commondatastorage.googleapis.com/books1000/notMNIST_large.tar.gz 
(большой набор данных);

- https://commondatastorage.googleapis.com/books1000/notMNIST_small.tar.gz 
(маленький набор данных).

Описание данных на английском языке доступно по ссылке:
http://yaroslavvb.blogspot.sg/2011/09/notmnist-dataset.html

### Задание 1.
Реализуйте полносвязную нейронную сеть с помощью библиотеки Tensor Flow. В качестве алгоритма оптимизации можно использовать, например, стохастический градиент (Stochastic Gradient Descent, SGD). Определите количество скрытых слоев от 1 до 5, количество нейронов в каждом из слоев до нескольких сотен, а также их функции активации (кусочно-линейная, сигмоидная, гиперболический тангенс и т.д.).

### Задание 2.
Как улучшилась точность классификатора по сравнению с логистической регрессией?
### Задание 3.
Используйте регуляризацию и метод сброса нейронов (dropout) для борьбы с переобучением. Как улучшилось качество классификации?
### Задание 4.
Воспользуйтесь динамически изменяемой скоростью обучения (learning rate). Наилучшая точность, достигнутая с помощью данной модели составляет 97.1%. Какую точность демонстрирует Ваша реализованная модель?

### Результаты
В качестве данных было выбрано большой набор данных
.
##### Отображение несколько изображений из набора данных.


#### гистограмма зависимости классов от количество изображений


#### График зависимости точности от эпох


