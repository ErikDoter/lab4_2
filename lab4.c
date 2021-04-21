#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>
// Граничные условия
// Левая граница
 #define LEFT_PERVIY 100 // ГУ первого рода
//#define LEFT_VTOROY 100 // ГУ второго рода
//#define LEFT_TRETII // ГУ третьего рода
// Правая граница
#define RIGHT_PERVIY 30
 //#define RIGHT_VTOROY 30
// #define RIGHT_TRETII
#define L 20000
#define at 2
#define dx 1
#define dt 0.1
#define time 0.2
#define init_temp 10 // Начальное значение температуры на стержне
const int xx = L/dx;
const int tt = time/dt;
#define MAKE_GNUPLOT 1 // Писать или нет в gnuplot файл (0, 1)
// Вычисление граничных условий
void calculate_border_conditions(double *T);
// Центральная разность
double central_difference(double t_left, double t_mid, double t_right);
// Вычисляем значение температуры на левой границе стержня. Передаёмследующий за первым элемент
double left_border(double next_node);
// Вычисляем значение температуры на правой границе стержня. Передаёмпредыдущий от последнего элемент
double right_border(double prev_node);
// Обмен граничными значениями температур между соседними процессами
void values_exchange(int rank, int total, double *T, double *stripe, int stripe_size, double *left, double *right);
// Вычисление температуры процессом
void calculate_temperature(int rank, int total, double *stripe_old, double *stripe_new, double *tmp, int stripe_size, double left, double right);
// Печать ленты
//void print_stripe(double *stripe, int stripe_size);
void create_gnuplot_config(); // Создаёт конфигруационный файлgnuplot
void write_data_to_file(FILE *f, double *T); // Пишет данные из T в файл
int main(int argc, char **argv)
{
    // Проверка параметров явной разностной схемы на устойчивость
    // Если при заданных параметрах схема неустойчива, результат вычислений будет неверным
/* if( dt > (pow(dx, 2)/(2*at)) )*/
/* {*/
/* fprintf(stderr, "With these values of dt, dx, at the result is
inaccurate!\n");*/
/* }*/
    // Инициализируем коммуникационные средства MPI
    MPI_Init(&argc, &argv);
    // Определение общего количества параллельных процессов в группе
    int total;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    // Проверка, кратно ли количество потоков количеству узлов
    if(xx % total != 0)
    {
        printf("The number of nodes is not a multiple of the number of processes!\n");
        MPI_Finalize();
    }
    // Определение идентификатора процесса в группе
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Указатель на массив температур
    double *T;
    // Выделяем память под него в корневом процессе и задаём начальнуютемпературу на стержне
    if(!rank)
    {

        if( dt > (pow(dx, 2)/(2*at)) )
        {
            fprintf(stderr, "With these values of dt, dx, at the result is inaccurate!\n");
        }

        T = (double *)malloc(sizeof(double) * xx);
        for(int i = 0; i < xx; ++i)
        {
            T[i] = init_temp;
        }
        T[0] = left_border(T[1]);
        T[xx-1] = right_border(T[xx-2]);
    }

    // Выделяем массив под "ленты"(участки стержня, которые будут рассчитываться каждым процессом)
    int stripe_size = xx / total;
    // "Старое" значение температуры. Используется при вычислении температуры в новый момент времени
    double *stripe_old = (double *)malloc(sizeof(double) * stripe_size);
    double *stripe_new = (double *)malloc(sizeof(double) * stripe_size);
    // Заполняем ленты начальными значениями из корневого процесса.Потом надо будет только пересылать
    // крайние значения
    MPI_Scatter(T, stripe_size, MPI_DOUBLE, stripe_old, stripe_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Чтобы рассчитать значение температуры на границе ленты, нам нужнознать соседнее с ней значение температуры
    // из другой ленты
    double left_from_stripe, right_from_stripe;
    // Если MAKE_GNUPLOT не равен нулю, то создаём файл, в который будем записывать данные
    // на каждой итерации и конфигурационный файл gnuplot
#if MAKE_GNUPLOT
    FILE *f;
    if(!rank)
    {
        create_gnuplot_config();
        f = fopen("plotting_data.dat", "w");
        if(f == NULL)
        {
            printf("Can't open/create plotting_data.dat file!\n");
        }
    }
#endif
    // Создаём структуры для времени
    struct timeval begin, end;
    if(!rank)
        gettimeofday(&begin, NULL);

    double *tmp;
    // Вычисляем значения для каждого временного слоя
    for(int i = 0; i < tt; ++i)
    {
        printf("aaaa\n");
        values_exchange(rank, total, T, stripe_old, stripe_size,
                        &left_from_stripe, &right_from_stripe);
        calculate_temperature(rank, total, stripe_old, stripe_new, tmp,
                              stripe_size, left_from_stripe, right_from_stripe);
        // Если флаг MAKE_GNUPLOT не равен нулю, то собираем данныекорневым процессом на каждой итерации и ведём запись
        // в gnuplot файл
#if MAKE_GNUPLOT
        MPI_Gather(stripe_new, stripe_size, MPI_DOUBLE, T, stripe_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if(!rank)
        {
            write_data_to_file(f, T);
        }
#endif
    }
    MPI_Gather(stripe_new, stripe_size, MPI_DOUBLE, T, stripe_size,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Вычисляем время работы программы
    double elapsed;
    if(!rank)
    {
        gettimeofday(&end, NULL);
        elapsed = (end.tv_sec - begin.tv_sec) + ((end.tv_usec -
                                                  begin.tv_usec)/1000000.0);
        printf("Processes %3d | Time, ms: %.3f\n", total, elapsed);
    }
    // Освобождаем память
#if MAKE_GNUPLOT
    if(!rank)
    {
        fclose(f);
        free(T);
    }
#endif
    free(stripe_old);
    free(stripe_new);
    MPI_Finalize();
}
void values_exchange(int rank, int total, double *T, double *stripe, int stripe_size, double *left, double *right)
{
    //MPI_Scatter(T, stripe_size, MPI_DOUBLE, stripe, stripe_size,MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Status stat;
    // Обмен крайними элементами лент с соседними процессами
    if(rank < total-1)
    {
        MPI_Sendrecv(&stripe[stripe_size - 1], 1, MPI_DOUBLE, rank+1, 0,
                     right, 1, MPI_DOUBLE, rank+1, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &stat);
    }
    if(rank > 0)
    {
        MPI_Sendrecv(&stripe[0], 1, MPI_DOUBLE, rank-1, 0,
                     left, 1, MPI_DOUBLE, rank-1, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &stat);
    }
}
void calculate_temperature(int rank, int total, double *stripe_old,
                           double *stripe_new, double *tmp, int stripe_size, double left, double
                           right)
{
    if(total > 1)
    {
        // Все процессы кроме первого (который обрабатывает левый край стержня)
        if(rank > 0)
        {
            // Вычисляем температуру на левой границе ленты
            stripe_new[0] = central_difference(left, stripe_old[0],
                                               stripe_old[1]);
            // Если это последний процесс (который обрабатывает правый край стержня), то
            // ещё и вычисляем условие на правой границе
            if(rank == total-1)
            {
                stripe_new[stripe_size-1] = right_border(stripe_old[stripe_size-2]);
            }
        }
        if(rank < total-1)
        {
            // Вычисляем температуру на правой границе ленты
            stripe_new[stripe_size-1] =
                    central_difference(stripe_old[stripe_size-2], stripe_old[stripe_size-1],
                                       right);
            if(rank == 0)
            {
                stripe_new[0] = left_border(stripe_old[1]);
            }
        }
    }
    else
    { // Если у нас только один процесс, то лента тоже одна
        stripe_new[0] = left_border(stripe_old[1]);
        stripe_new[stripe_size-1] = right_border(stripe_old[stripe_size]);
    }
    // Вычисляем значение температуры для середины ленты
    printf("%f, %f, %f\n", stripe_old[0], stripe_old[1], stripe_old[2]);
    for(int i = 1; i < stripe_size-1; ++i)
    {
        stripe_new[i] = central_difference(stripe_old[i-1],
                                           stripe_old[i], stripe_old[i+1]);
    }
    // Копируем только что вычисленные значения в массив stripe_old
    memcpy(stripe_old, stripe_new, sizeof(double)*stripe_size);
    //stripe_new = tmp;
}
// t_left -- Tj-1; t_mid -- Tj; t_right -- Tj+1
double central_difference(double t_left, double t_mid, double t_right)
{
    // Возвращает значение температуры в следующий момент времени не на границе стержня
    return at*((t_right - 2*t_mid + t_left)*dt)/pow(dx, 2) + t_mid;
}
double left_border(double next_node)
{
    double result = 0; // Значение на левой границе
    // Расчёт левой границы
    // ГУ первого рода
#ifdef LEFT_PERVIY
    result = LEFT_PERVIY;
#endif
    // ГУ второго рода
#ifdef LEFT_VTOROY
    result = ((double)LEFT_VTOROY * dx)/at + next_node;
#endif
    // ГУ третьего рода
#ifdef LEFT_TRETII
    // Уравнение третьего рода: at * (dT/dx) = F(T),
    // где F(T) = C1*T + C2. C1, C2 -- некоторые константы
    const double C1 = 0.5;
    const double C2 = 5;
    result = (C2*dx + next_node*at)/(at - C1*dx);
#endif
    return result;
}
double right_border(double prev_node)
{
    double result;
    // РасчЁт правой границы
    // ГУ первого рода
#ifdef RIGHT_PERVIY
    result = RIGHT_PERVIY;
#endif
    // ГУ второго рода
#ifdef RIGHT_VTOROY
    result = ((double)RIGHT_VTOROY * dx)/at + prev_node;
#endif
    // ГУ третьего рода
#ifdef RIGHT_TRETII
    // Уравнение третьего рода: at * (dT/dx) = F(T),
 // где F(T) = C1*T + C2. C1, C2 -- некоторые константы
 const double C1 = 0.5;
 const double C2 = 5;
 result = (C2*dx + prev_node*at)/(at - C1*dx);
#endif
    return result;
}
/*void print_stripe(double *stripe, int stripe_size)*/
/*{*/
/* for(int i = 0; i < stripe_size; ++i)*/
/* {*/
/* printf("%.2f ", stripe[i]);*/
/* }*/
/* printf("\n");*/
/*}*/

void create_gnuplot_config()
{
    FILE* f = fopen("gnu.config", "w");
    if(f == NULL)
    {
        fprintf(stderr, "Cannot open/create gnu.config file!\n");
        return;
    }
    fprintf(f, "set terminal gif animate delay 10\nset output 'foobar.gif'\n");
    fprintf(f, "set style line 1 \\\nlinecolor rgb '#0060ad' \\\nlinetype 1 linewidth 2 \\\n\n");
    fprintf(f, "do for [i=0:%d] {\n", tt-1);
    fprintf(f, "plot 'plotting_data.dat' index i with lines linestyle 1\n");
    fprintf(f, "pause %f\n}", (double)5./tt);
    fclose(f);
}

void write_data_to_file(FILE *f, double *T)
{
    for(int i = 0; i < xx; ++i)
    {
        // Пишем два столбца(по оси X и Y). По оси X узлы стержня, по Y -- значение температуры
                fprintf(f, "%d %f\n", i, T[i]);
    }
    fprintf(f, "\n\n");
}