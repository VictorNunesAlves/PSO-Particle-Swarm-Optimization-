# Curso de Timetabling usando Constriction Particle Swarm Optimization (PSO) com Busca Local

Este projeto implementa uma **metaheurística PSO (Particle Swarm Optimization)** para resolver o problema de **alocação de horários de cursos universitários**, inspirado no artigo:

> *Solving University Course Timetabling Problems Using Constriction Particle Swarm Optimization with Local Search*,
> Ruey-Maw Chen e Hsiao-Fang Shih, National Chinyi University of Technology, Taiwan.

---

## Estrutura do Problema

O problema consiste em **alocar 20 cursos em horários, salas e professores**, considerando restrições rígidas e preferências:

* **Cursos:** 20 cursos, cada um definido por:

  ```json
  {
      "course_id": "COURSE1",
      "teacher": "T7",
      "class": "C4",
      "room": "R8",
      "duration": 2
  }
  ```
* **Recursos:**

  * Professores: 16
  * Turmas: 10
  * Salas: 10
  * Horários: 20 (4 por dia em 5 dias)
* **Objetivo:** Atribuir cada curso a um horário e sala sem conflitos de professor, turma ou horário.

---

## Representação da Partícula

Cada **partícula** no PSO representa um **possível cronograma**:

```text
(ID do curso, Professor, Turma, Sala, Horário inicial, Duração)
Exemplo:
('COURSE1', 'T1', 'C4', 'R3', 5, 2)
('COURSE2', 'T2', 'C4', 'R10', 12, 3)
...
('COURSE20', 'T3', 'C6', 'R9', 0, 2)
```

* Cada posição da partícula representa o **horário inicial do curso**.
* A velocidade determina como essa posição muda a cada iteração.

---

## Restrições Consideradas

1. Um **professor** leciona apenas uma turma por horário.
2. Uma **turma** participa de apenas um curso por horário.
3. Uma **sala** contém apenas um curso por horário.
4. Cursos de **3 horas seguidas** devem estar no mesmo dia sem interrupções de almoço.
5. Professores devem lecionar **pelo menos duas vezes na semana**.
6. Preferências de professores e alunos são levadas em consideração para aumentar a qualidade do cronograma.

---

## Funcionamento do PSO

1. **Inicialização:**

   * Criam-se várias partículas com horários aleatórios.
   * Cada partícula é avaliada por uma **função de fitness** que combina pontuação de preferências e penalidades por conflitos.

2. **Atualização de Partículas:**

   * Cada partícula atualiza sua posição com base na experiência própria (**pBest**) e na melhor global (**gBest**):

   $$
   $$

v\_{i}^{t+1} = k \cdot w \cdot (v\_i^t + c\_1 r\_1 (pBest\_i - x\_i^t) + c\_2 r\_2 (gBest - x\_i^t))
]

$$
x_i^{t+1} = x_i^t + v_i^{t+1}
$$

* $v_i$ = velocidade da partícula
* $x_i$ = posição atual (horário inicial)
* $w$ = fator de inércia
* $c_1, c_2$ = coeficientes cognitivo e social
* $r_1, r_2$ = números aleatórios entre 0 e 1
* $k$ = fator de constrição (melhora convergência)

3. **Busca Local:**

   * Após atualizar posições, é aplicado um **intercâmbio local** entre horários para melhorar o fitness da partícula.
   * Se a solução local melhorar o fitness, a partícula adota essa solução.

4. **Iteração:**

   * O processo se repete por várias iterações até convergir para o **melhor cronograma possível**.

---

## Função de Fitness

A função de fitness considera:

* Pontuação de **preferências** de professores e turmas.
* Penalidades para:

  * Conflitos de professor, turma ou sala.
  * Horários bloqueados.
  * Cursos que não podem ser alocados devido à duração.

A **melhor partícula** (gbest) representa o cronograma com **menor conflito e maior satisfação de preferências**.

---

## Resultados

O código permite comparar diferentes configurações de PSO:

* Diferentes valores de $c_1, c_2$ (coeficientes cognitivo e social)
* Diferentes valores de $w$ (inércia)
* Uso ou não da busca local

O desempenho é visualizado em gráficos de **fitness x iterações**.

---

## Código de Exemplo

O projeto utiliza **Python**, `numpy`, `matplotlib` e `collections` para implementar PSO com busca local, geração aleatória de cursos e cálculo de fitness.

* Inicialização de partículas
* Avaliação de fitness
* Atualização de velocidade e posição
* Busca local com intercâmbio de horários
* Registro histórico do melhor fitness

---

## Conclusão

O uso do **Constriction PSO com busca local** permite gerar **cronogramas universitários válidos** respeitando restrições rígidas e preferências, mostrando que **metaheurísticas baseadas em enxames** são eficazes para problemas de otimização combinatória complexos.
