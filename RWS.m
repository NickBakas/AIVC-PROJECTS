% 1: Initialize a population of chromosomes
% 2: Evaluate each chromosome in the population
% 3: Create new chromosomes by mating current chromosomes.Apply crossover and mutation operators.
% 4: Delete less fit members of the population.
% 5: Evaluate and insert the new chromosomes in the population
% 6: If stopping criterion is satisfied, then stop and return
% the best chromosome; otherwise, go to step 3.
% END GA

clc;
clear all;

tic;
%1 ΣΥΝΟΛΟ ΠΟΛΕΩΝ
CitysumA=100; 
populationSize=50;
maxgen=1000;


%2 ΑΡΧΙΚΟΠΟΙΗΣΗ ΜΕΤΡΙΚΩΝ
bestFitnessPerGeneration = zeros(maxgen, 1);
worstFitnessPerGeneration = zeros(maxgen, 1);
averageFitnessPerGeneration = zeros(maxgen, 1);
bestTotalDistancesPerGeneration = zeros(maxgen, 1);
worstTotalDistancesPerGeneration = zeros(maxgen, 1);
averageTotalDistancesPerGeneration = zeros(maxgen, 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%3 ΤΥΧΑΙΕΣ ΑΠΟΣΤΑΣΕΙΣ ΑΠΟ (0,100), ΣΥΜΜΕΤΡΙΑ ΚΑΙ 0 ΣΤΗ ΔΙΑΓΩΝΙΟ
distances = randi([1, 100], CitysumA, CitysumA);
distances = triu(distances, 1) + triu(distances, 1)';
distances = distances - diag(distances);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%4 ΑΡΧΙΚΟΠΟΙΗΣΗ ΠΛΗΘΥΣΜΟΥ
for i = 1:populationSize
    Population(i, :) = randperm(CitysumA);    
end

disp('Distance Matrix:');
disp(distances);
disp('Population Matrix:');
disp(Population);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for gen = 1:maxgen

    %% ΑΡΧΙΚΟΠΟΙΗΣΗ ΜΕΤΑΒΛΗΤΗΣ ΓΙΑ ΜΕΤΑΦΡΑΣΗ ΑΠΟΣΤΑΣΕΩΝ ΣΕ ΚΩΔΙΚΟ POPULATION
    %5 Η ΜΕΤΑΒΛΗΤΗ ΘΑ ΒΟΗΘΗΣΕΙ ΣΤΟ ΑΘΡΟΙΣΜΑ ΑΠΟΣΤΑΣΕΩΝ ΓΙΑ ΤΟ FITNESS
    calculatedDistances = zeros(size(Population, 1), size(Population, 2) );
    for i = 1:size(Population, 1)
        for j = 1:size(Population, 2)  
            if j~=size(Population, 2)
            cityIndex1 = Population(i, j);
            cityIndex2 = Population(i, j + 1);
            calculatedDistances(i, j) = distances(cityIndex1, cityIndex2);
            else
            cityIndex1 = Population(i, j);
            cityIndex2 = Population(i,1);  
            calculatedDistances(i, j) = distances(cityIndex1, cityIndex2);
            end
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% ΥΠΟΛΟΓΙΣΜΟΣ FITNESS ΩΣ 1/TOTALDISTANCE
    %6 ΥΠΟΛΟΓΙΣΜΟΣ ΠΙΘΑΝΟΤΗΤΑΣ
    totalDistances = sum(calculatedDistances, 2);
    disp(totalDistances)
    Fitness = zeros(populationSize, 1);
    for i = 1:populationSize
        Fitness(i) = 1 / totalDistances(i);
    end
    disp('Fitness Matrix:');
    disp(Fitness);
    
    totalFitness=sum(Fitness,1);
    Probability=zeros(size(Fitness));
    
    for i=1:size(Fitness,1)
        Probability(i)=Fitness(i)/totalFitness;
    end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ΔΗΜΙΟΥΡΓΙΑ RANGE ΡΟΥΛΕΤΑΣ ΜΕ ΒΟΗΘΕΙΑ ΠΙΘΑΝΟΤΗΤΑΣ
%range για δείκτη και αρίθμηση cumsum
    SummedProbability = cumsum(Probability);
    ranges = zeros(populationSize, 1);
    rangeNumbers = (1:populationSize);
    
    for i = 1:populationSize
        ranges(i, 1) = (rangeNumbers(i));
        ranges(i, 2) = SummedProbability(i);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
chr=zeros(2,size(Population,2));
    while chr(1,:)==chr(2,:)
        for u=1:2
            randomFit = rand;
            disp(randomFit);
            for i = 1:length(SummedProbability)
                check=SummedProbability(i);
                
                        if randomFit>SummedProbability(i) && randomFit<=SummedProbability(i+1)
                           chr(u, :)=Population(i+1,:);
                        elseif randomFit<=SummedProbability(1) && randomFit>0
                             chr(u, :)=Population(1,:);
                        end
            end
        end
        u=u-1;

    end
    disp(SummedProbability);
    disp(chr);          



 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% ORDER CROSSOVER & SWAP MUTATION
    %7
    [off1, off2] = Crossover_Order(chr(1, :), chr(2, :));
    disp('Offspring 1 after Ordered Crossover:');
    disp(off1);
    disp('Offspring 2 after Ordered Crossover:');
    disp(off2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
    %% χαμηλό mutation για προστασία δυνητικά καλού χρωμοσώματος
    mutated = 10; 
    off1 = Mutation_Swap(off1, mutated);
    off2 = Mutation_Swap(off2, mutated);
    disp('Offspring 1 after Mutation:');
    disp(off1);
    disp('Offspring 2 after Mutation:');
    disp(off2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    %% Υπολογισμός απόστασης απο πόλη σε πόλη με δείκτες y1,y2
    %9-10
    off1Distances = zeros(1, CitysumA);
    for i = 1:CitysumA
        cityIndex1 = off1(i);
        if i < CitysumA
            cityIndex2 = off1(i + 1);
        else
            cityIndex2 = off1(1);
        end
        off1Distances(i) = distances(cityIndex1, cityIndex2);
    end
    
    off2Distances = zeros(1, CitysumA);
    for i = 1:CitysumA
        cityIndex1 = off2(i);
        if i < CitysumA
            cityIndex2 = off2(i + 1);
        else
            cityIndex2 = off2(1);
        end
        off2Distances(i) = distances(cityIndex1, cityIndex2);
    end
    
    newOffspringFitness1 = 1 / sum(off1Distances);
    newOffspringFitness2 = 1 / sum(off2Distances);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    %% Index και αντικατάσταση των χειρότερων χρωμοσωμάτων με τα νέα
    [sortedFitness,sortedIdx] = sort(Fitness,'ascend');
    lowestFitness = sortedIdx(1:2);

    
    if Fitness(sortedIdx(1))<newOffspringFitness1
       Population(lowestFitness(1), :) = off1;
    end

    if Fitness(sortedIdx(2))<newOffspringFitness2
       Population(lowestFitness(2), :) = off2;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %11 METRICS
    averageFitnessPerGeneration(gen) = mean(Fitness);
    bestFitnessPerGeneration(gen) = max(Fitness);
    worstFitnessPerGeneration(gen) = min(Fitness);
    worstTotalDistancesPerGeneration(gen) = max(totalDistances);
    bestTotalDistancesPerGeneration(gen) = min(totalDistances);
    averageTotalDistancesPerGeneration(gen) = mean(totalDistances);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 


%% Convergence Best, Average, Worst Fitness
%12 PLOT TA METRICS

figure;
plot(1:maxgen, bestFitnessPerGeneration, 'color',[0, 0.5, 0], 'LineWidth', 1);
hold on;
plot(1:maxgen, averageFitnessPerGeneration,'color',[0, 0, 0], 'LineWidth', 1);
plot(1:maxgen, worstFitnessPerGeneration, 'color',[0.5, 0, 0.5], 'LineWidth', 1);
xlabel('Generations');
ylabel('Fitness');
legend('Best Fitness', 'Average Fitness', 'Worst Fitness'); 
title('Convergence-Fitness/Generations');
grid on;
hold off;
%% Convergence Best, Average, Worst Distances
figure;
plot(1:maxgen, bestTotalDistancesPerGeneration, 'color',[0, 0.5, 0], 'LineWidth', 1);
hold on;
plot(1:maxgen, averageTotalDistancesPerGeneration, 'color',[0, 0, 0], 'LineWidth', 1);
plot(1:maxgen, worstTotalDistancesPerGeneration, 'color',[0.5, 0, 0.5],'LineWidth', 1);
xlabel('Generations');
ylabel('Distances');
legend('Best Total Distances', 'Average Total Distances', 'Worst Total Distances');
title('Convergence-Distances/Generations');
grid on;
hold off; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
toc;


%% ORDER CROSSOVER
function [off1, off2] = Crossover_Order(xx1, xx2)
    off1 = zeros(size(xx1));
    off2 = zeros(size(xx2));

    % Order Crossover (επιλογη χρηστη)
    crossoverStart = 20;
    crossoverEnd = 70;

    % Απο parent1 σε offspring2 & parent2 σε offspring1
    off2(crossoverStart:crossoverEnd) = xx1(crossoverStart:crossoverEnd);
    remainingValues2 = setdiff(xx2, off2, 'stable');
    remainingIndices2 = setdiff(1:numel(xx2), crossoverStart:crossoverEnd);
    off2(remainingIndices2) = remainingValues2;
    off1(crossoverStart:crossoverEnd) = xx2(crossoverStart:crossoverEnd);
    remainingValues1 = setdiff(xx1, off1, 'stable');
    remainingIndices1 = setdiff(1:numel(xx1), crossoverStart:crossoverEnd);
    off1(remainingIndices1) = remainingValues1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% MUTATION
function mutatedOffspring = Mutation_Swap(offspring, mutated)
    mutatedOffspring = offspring; 
    Swap = sort(randperm(numel(offspring), mutated));
    mutatedOffspring(Swap) = mutatedOffspring(fliplr(Swap));
end
