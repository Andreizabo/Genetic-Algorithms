#pragma warning(disable: 4996)
#include <vector>
#include <iostream>
#include <thread>
#include <cmath>
#include <sstream>
#include <cstring>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <iomanip>
#include <conio.h>
#include <algorithm>
#include <random>
#include <chrono>
#include <direct.h>

#define PI 3.14159265358979323846
#define M 10

using namespace std;

class Chromosome {
public:
	vector<char> bits;
	double fitness;
	long bitsSize;

	Chromosome(long bitsSizeArg) {
		bitsSize = bitsSizeArg;
		bits.reserve(bitsSize);
		fitness = 0;
	}
};

double a = -500, b = 500; //capete interval
int prec = 5; //precizie;
unsigned int n; // nr de dimensiuni
int l; //length
int POPULATION_SIZE = 1000;
int GENERATIONS = 10000;
double MUTATION_PROB = 0.01;
double CROSSOVER_PROB = 0.25;

//FUNCTIONS

double deJong(std::vector<double>& v) {
	double result = 0;
	for (double d : v) {
		result += (d * d);
	}
	return result;
}
double deJongFitness(double value) {
	return pow(value, -10); //USE POPULATION 1000 AND MUTATION_CHANCE 0.01 FOR BEST RESULTS
}

double schwefel(std::vector<double>& v) {
	double result = 0;
	for (double d : v) {
		result += (d * sin(sqrt(abs(d))) * -1);
	}
	return result;
}
double schwefelFitness(double value) {
	return pow(1.01, atan2(1e-4, value) * abs(value)); //USE GENERATIONS 2000 AND MUTATION_CHANCE 0.02 FOR BEST RESULTS
}

double rastrigin(std::vector<double>& v) {
	double result = 10 * v.size();
	for (double d : v) {
		result += ((d * d) - (10 * cos(2 * PI * d)));
	}
	return result;
}
double rastriginFitness(double value) {
	return pow(value, -10);
	//BEST: MUTATION 0.01, CROSSOVER 0.05 GENERATIONS 10000 -10
	//BEST: MUTATION 0.01, CROSSOVER 0.05 GENERATIONS 10000 -25
	//BEST: MUTATION 0.01, CROSSOVER 0.2 GENERATIONS 10000 -10
	//BEST: MUTATION 0.0125, CROSSOVER 0.25 GENERATIONS 12500 -10 <- I used this
}

double michalewicz(std::vector<double>& v) {
	double result = 0;
	int size = v.size();
	for (int i = 1; i <= size; ++i) {
		result += (sin(v[i - 1]) * pow(sin((i * v[i - 1] * v[i - 1]) / PI), M * 2));
	}
	result *= -1;
	return result;
}
double michalewiczFitness(double value) {
	return pow(1.05, pow(value, 2));
}

//END OF FUNCTIONS

int length(double a, double b, int prec)
{
	return ceil(log((b - a) * pow(10, prec)) / log(2));
}

double decodeDimension(vector<char>::iterator itStart, vector<char>::iterator itEnd, int l, double a, double b)
{
	unsigned long bi = 0;
	for (auto i = itStart; i != itEnd; ++i)
	{
		bi *= 2;
		bi += *i;
	}

	double s = bi / (pow(2, l) - 1);
	return s * (b - a) + a;

}

vector<double> decode(vector<char>& bits, int l, unsigned int n, double a, double b)
{
	vector<double> ret;
	vector<char>::iterator itStart, itEnd;
	for (int i = 0; i < n; ++i) {
		itStart = bits.begin() + i * l;
		itEnd = itStart + l;

		double x = decodeDimension(itStart, itEnd, l, a, b);
		ret.push_back(x);
	}
	return ret;
}

void select(vector<Chromosome>& population) {
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	mt19937 myRand(seed);
	uniform_real_distribution<double> realDistribution(0.0, 1.0);

	int bitsSize = population[0].bitsSize;

	double totalFitness = 0;
	for (int i = 0; i < POPULATION_SIZE; ++i) {
		totalFitness += population[i].fitness;
	}

	vector<double> individualProbabilities, accumulatedProbabilities;
	individualProbabilities.reserve(POPULATION_SIZE);
	accumulatedProbabilities.reserve(POPULATION_SIZE + 1);
	accumulatedProbabilities.push_back(0);

	for (int i = 0; i < POPULATION_SIZE; ++i) {
		individualProbabilities.push_back(population[i].fitness / totalFitness);
		accumulatedProbabilities.push_back(accumulatedProbabilities.back() + individualProbabilities.back());
	}

	vector<Chromosome> newPopulation;
	newPopulation.reserve(POPULATION_SIZE);
	for (int i = 0; i < POPULATION_SIZE; ++i) {
		double auxRandom = realDistribution(myRand);
		for (int j = 0; j < POPULATION_SIZE; ++j) {
			if (accumulatedProbabilities[j] < auxRandom && accumulatedProbabilities[j + 1] >= auxRandom) {
				newPopulation.push_back(population[j]);
				break;
			}
		}
	}

	population = newPopulation;
}

void mutate(vector<Chromosome>& population) {
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	mt19937 myRand(seed);
	uniform_real_distribution<double> realDistribution(0.0, 1.0);

	for (int i = 0; i < POPULATION_SIZE; ++i) {
		for (int j = 0; j < population[i].bitsSize; ++j) {
			double auxRandom = realDistribution(myRand);
			if (auxRandom < MUTATION_PROB) {
				population[i].bits[j] = !population[i].bits[j];
			}
		}
	}
}

void crossOver(vector<Chromosome>& population) {
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	mt19937 myRand(seed);
	uniform_real_distribution<double> realDistribution(0.0, 1.0);

	vector<pair<double, int>> crossoverProbabilities;
	crossoverProbabilities.reserve(POPULATION_SIZE);

	for (int i = 0; i < POPULATION_SIZE; ++i) {
		crossoverProbabilities.emplace_back(realDistribution(myRand), i);
	}
	sort(crossoverProbabilities.begin(), crossoverProbabilities.end());
	for (int i = 1; i < POPULATION_SIZE && crossoverProbabilities[i].first < CROSSOVER_PROB; i += 2) {
		seed = chrono::system_clock::now().time_since_epoch().count();
		mt19937 myIntRand(seed);
		uniform_int_distribution<int> intDistribution(1, population[i].bitsSize - 2);
		int cutPoint = intDistribution(myIntRand);
		for (int j = 0; j < cutPoint; ++j) {
			swap(population[crossoverProbabilities[i].second].bits[j], population[crossoverProbabilities[i - 1].second].bits[j]);
		}
	}
}

double evaluate(vector<Chromosome>& population, double (*func)(vector<double>&), double (*fitness)(double)) {
	double auxMinima = 1000000;

	for (int i = 0; i < POPULATION_SIZE; ++i) {
		vector<double> decoded = decode(population[i].bits, l, n, a, b);
		double value = func(decoded);
		population[i].fitness = fitness(value);

		auxMinima = min(auxMinima, value);
	}

	return auxMinima;
}


void genetics(unsigned int n, int l, int bitsSize, double a, double b, double (*func)(vector<double>&), double (*fitness)(double), ofstream& f) {
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	mt19937 myRand(seed);
	clock_t start, end;
	start = clock();
	vector<char> bits;
	vector<Chromosome> population(POPULATION_SIZE, Chromosome(bitsSize));
	for (int i = 0; i < POPULATION_SIZE; ++i) {
		bits.clear();

		for (int j = 0; j < population[i].bitsSize; ++j) {
			population[i].bits.push_back(myRand() % 2);
		}
		
		vector<double> decoded = decode(population[i].bits, l, n, a, b);
		population[i].fitness = fitness(func(decoded));
	}

	double globalMinima = 1000000;
	for (int generation = 0; generation < GENERATIONS; ++generation) {
		select(population);
		mutate(population);
		crossOver(population);
		globalMinima = min(globalMinima, evaluate(population, func, fitness));
	}

	double minima = 1000000;
	for (int i = 0; i < POPULATION_SIZE; ++i) {
		vector<double> decoded = decode(population[i].bits, l, n, a, b);
		double value = func(decoded);
		minima = min(minima, value);
	}

	end = clock();

	f << minima << '\n' << (double)((double)(end - start) / (double)CLOCKS_PER_SEC) << '\n';
}

int getIntId(thread::id id) {
	stringstream buffer;
	buffer << id;
	return stoull(buffer.str());
}

void mainFunctionGenetics(int size, double (*func)(vector<double>&), double (*fitness)(double), clock_t beginning, const char* function) {
	n = size;
	int id = getIntId(this_thread::get_id());
	char* fileName = new char[256];
	strcpy(fileName, function);
	strcat(fileName, "/");
	strcat(fileName, to_string(size).c_str());
	strcat(fileName, "_");
	strcat(fileName, to_string(POPULATION_SIZE).c_str());
	strcat(fileName, "_");
	strcat(fileName, to_string(GENERATIONS).c_str());
	strcat(fileName, "_");
	strcat(fileName, to_string(MUTATION_PROB).c_str());
	strcat(fileName, "_");
	strcat(fileName, to_string(CROSSOVER_PROB).c_str());
	strcat(fileName, "_Thread_");
	strcat(fileName, to_string(id).c_str());
	strcat(fileName, "_");
	strcat(fileName, to_string(clock() - beginning).c_str());
	strcat(fileName, ".txt");
	ofstream aux(fileName);
	int L = l * size;
	genetics(size, l, L, a, b, func, fitness, aux);
	aux.close();
}

void solve(int aArg, int bArg, int precisionArg, double (*func)(vector<double>&), double (*fitness)(double), const char* function, int pop, int gen, double mut, double xover) {
	a = aArg;
	b = bArg;
	prec = precisionArg;
	l = length(a, b, prec);

	POPULATION_SIZE = pop;
	GENERATIONS = gen;
	MUTATION_PROB = mut;
	CROSSOVER_PROB = xover;

	char* dirName = new char[256];
	strcpy(dirName, "./");
	strcat(dirName, function);
	_mkdir(dirName);

	clock_t start, end;
	start = clock();
	cout << "\nStarting analysis of " << function << "\n";
	cout << "Starting threads for size 5\n";
	for (int i = 0; i < 3; ++i) { //running 36 times
		thread t1(mainFunctionGenetics, 5, func, fitness, start, function);
		thread t2(mainFunctionGenetics, 5, func, fitness, start, function);
		thread t3(mainFunctionGenetics, 5, func, fitness, start, function);
		thread t4(mainFunctionGenetics, 5, func, fitness, start, function);
		thread t5(mainFunctionGenetics, 5, func, fitness, start, function);
		thread t6(mainFunctionGenetics, 5, func, fitness, start, function);
		thread t7(mainFunctionGenetics, 5, func, fitness, start, function);
		thread t8(mainFunctionGenetics, 5, func, fitness, start, function);
		thread t9(mainFunctionGenetics, 5, func, fitness, start, function);
		thread t10(mainFunctionGenetics, 5, func, fitness, start, function);
		thread t11(mainFunctionGenetics, 5, func, fitness, start, function);
		thread t12(mainFunctionGenetics, 5, func, fitness, start, function);

		t1.join();
		t2.join();
		t3.join();
		t4.join();
		t5.join();
		t6.join();
		t7.join();
		t8.join();
		t9.join();
		t10.join();
		t11.join();
		t12.join();

		cout << "Finished iteration " << i << " for size 5\n";
	}

	cout << "Starting threads for size 10\n";
	for (int i = 0; i < 3; ++i) { //running 36 times
		thread t1(mainFunctionGenetics, 10, func, fitness, start, function);
		thread t2(mainFunctionGenetics, 10, func, fitness, start, function);
		thread t3(mainFunctionGenetics, 10, func, fitness, start, function);
		thread t4(mainFunctionGenetics, 10, func, fitness, start, function);
		thread t5(mainFunctionGenetics, 10, func, fitness, start, function);
		thread t6(mainFunctionGenetics, 10, func, fitness, start, function);
		thread t7(mainFunctionGenetics, 10, func, fitness, start, function);
		thread t8(mainFunctionGenetics, 10, func, fitness, start, function);
		thread t9(mainFunctionGenetics, 10, func, fitness, start, function);
		thread t10(mainFunctionGenetics, 10, func, fitness, start, function);
		thread t11(mainFunctionGenetics, 10, func, fitness, start, function);
		thread t12(mainFunctionGenetics, 10, func, fitness, start, function);

		t1.join();
		t2.join();
		t3.join();
		t4.join();
		t5.join();
		t6.join();
		t7.join();
		t8.join();
		t9.join();
		t10.join();
		t11.join();
		t12.join();

		cout << "Finished iteration " << i << " for size 10\n";
	}

	cout << "Starting threads for size 30\n";
	for (int i = 0; i < 3; ++i) { //running 36 times
		thread t1(mainFunctionGenetics, 30, func, fitness, start, function);
		thread t2(mainFunctionGenetics, 30, func, fitness, start, function);
		thread t3(mainFunctionGenetics, 30, func, fitness, start, function);
		thread t4(mainFunctionGenetics, 30, func, fitness, start, function);
		thread t5(mainFunctionGenetics, 30, func, fitness, start, function);
		thread t6(mainFunctionGenetics, 30, func, fitness, start, function);
		thread t7(mainFunctionGenetics, 30, func, fitness, start, function);
		thread t8(mainFunctionGenetics, 30, func, fitness, start, function);
		thread t9(mainFunctionGenetics, 30, func, fitness, start, function);
		thread t10(mainFunctionGenetics, 30, func, fitness, start, function);
		thread t11(mainFunctionGenetics, 30, func, fitness, start, function);
		thread t12(mainFunctionGenetics, 30, func, fitness, start, function);

		t1.join();
		t2.join();
		t3.join();
		t4.join();
		t5.join();
		t6.join();
		t7.join();
		t8.join();
		t9.join();
		t10.join();
		t11.join();
		t12.join();

		cout << "Finished iteration " << i << " for size 30\n";
	}
	
	end = clock();
	cout << function << " took " << (double)((double)(end - start) / (double)(CLOCKS_PER_SEC)) << " seconds.\n";
}

int main() {
	//DEJONG
	solve(-5.12, 5.12, 5, deJong, deJongFitness, "DeJong", 200, 2500, 0.01, 0.2);
	//SCHWEFEL
	solve(-500, 500, 5, schwefel, schwefelFitness, "Schwefel", 1000, 2500, 0.02, 0.2);
	//Rastrigin
	solve(-5.12, 5.12, 5, rastrigin, rastriginFitness, "Rastrigin", 1000, 10000, 0.0125, 0.3);
	//Michalewicz
	solve(0, PI, 5, michalewicz, michalewiczFitness, "Michalewicz", 1000, 10000, 0.01, 0.25);

	getch();
	return 0;
}