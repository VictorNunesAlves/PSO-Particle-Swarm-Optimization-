import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt

def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

NUM_TEACHERS = 16
NUM_CLASSES = 10
NUM_ROOMS = 10
NUM_TIMESLOTS = 20  # 4 por dia * 5 dias
blocked_slots = {10, 11}  # quinta-feira à tarde

def generate_courses():
    courses = []
    for i in range(20):
        teacher = f"T{np.random.randint(1, NUM_TEACHERS + 1)}"
        class_id = f"C{np.random.randint(1, NUM_CLASSES + 1)}"
        room = f"R{np.random.randint(1, NUM_ROOMS + 1)}"
        duration = np.random.choice([2, 3])
        course_id = f"COURSE{i+1}"
        courses.append({
            "course_id": course_id,
            "teacher": teacher,
            "class": class_id,
            "room": room,
            "duration": duration
        })
    return courses

def random_preferences():
    def generate_pref():
        return np.random.choice([-10, 1, 2, 3, 4, 5], p=[0.1, 0.18, 0.18, 0.18, 0.18, 0.18])

    prefs_teacher = {
        f"T{i+1}": np.array([generate_pref() for _ in range(NUM_TIMESLOTS)])
        for i in range(NUM_TEACHERS)
    }
    prefs_class = {
        f"C{i+1}": np.array([generate_pref() for _ in range(NUM_TIMESLOTS)])
        for i in range(NUM_CLASSES)
    }
    return prefs_teacher, prefs_class

def is_valid_schedule(schedule):
    teacher_usage = defaultdict(set)
    class_usage = defaultdict(set)
    room_usage = defaultdict(set)

    for item in schedule:
        _, tid, clid, rid, start_ts, duration = item
        for t in range(start_ts, start_ts + duration):
            if t >= NUM_TIMESLOTS or t in blocked_slots:
                return False
            if t in teacher_usage[tid] or t in class_usage[clid] or t in room_usage[rid]:
                return False
            teacher_usage[tid].add(t)
            class_usage[clid].add(t)
            room_usage[rid].add(t)

    return True

def fitness_with_constraints(schedule, prefs_teacher, prefs_class):
    teacher_usage = defaultdict(set)
    class_usage = defaultdict(set)
    room_usage = defaultdict(set)

    total_score = 0
    penalty = 0

    for item in schedule:
        _, tid, clid, rid, start_ts, duration = item

        if start_ts == -1:
            penalty += 10
            continue

        for i in range(duration):
            ts = start_ts + i

            if ts >= NUM_TIMESLOTS:
                penalty += 10
                continue

            if ts in blocked_slots:
                penalty += 10

            if ts in teacher_usage[tid]:
                penalty += 10
            else:
                teacher_usage[tid].add(ts)

            if ts in class_usage[clid]:
                penalty += 10
            else:
                class_usage[clid].add(ts)

            if ts in room_usage[rid]:
                penalty += 10
            else:
                room_usage[rid].add(ts)

            t_pref = prefs_teacher[tid][ts]
            c_pref = prefs_class[clid][ts]

            if t_pref == -10:
                penalty += 10
            else:
                total_score += t_pref

            if c_pref == -10:
                penalty += 10
            else:
                total_score += c_pref

    return total_score - penalty

def build_schedule_from_position(courses, position):
    schedule = []
    for i, course in enumerate(courses):
        teacher = course["teacher"]
        class_id = course["class"]
        room = course["room"]
        duration = course["duration"]
        course_id = course["course_id"]
        start = int(round(position[i]))

        conflict = False
        for d in range(duration):
            ts = start + d
            if ts >= NUM_TIMESLOTS or ts in blocked_slots:
                conflict = True
                break

        if conflict:
            schedule.append((course_id, teacher, class_id, room, -1, duration))
        else:
            schedule.append((course_id, teacher, class_id, room, start, duration))

    return schedule

def interchange(schedule):
    new_schedule = schedule.copy()
    i, j = random.sample(range(len(new_schedule)), 2)

    if new_schedule[i][4] == -1 or new_schedule[j][4] == -1:
        return new_schedule

    a = list(new_schedule[i])
    b = list(new_schedule[j])
    a[4], b[4] = b[4], a[4]
    new_schedule[i] = tuple(a)
    new_schedule[j] = tuple(b)
    return new_schedule

def run_course_timetabling_pso(c1, c2, w, seed=40, num_particles=30, iterations=3000, use_local_search=True):
    set_random_seed(seed)

    courses = generate_courses()
    prefs_teacher, prefs_class = random_preferences()

    num_courses = len(courses)
    positions = [np.random.uniform(0, NUM_TIMESLOTS - 3, size=num_courses).tolist() for _ in range(num_particles)]
    velocities = [np.zeros(num_courses).tolist() for _ in range(num_particles)]

    pbests = positions.copy()
    pbests_scores = [fitness_with_constraints(build_schedule_from_position(courses, p), prefs_teacher, prefs_class) for p in positions]
    gbest = pbests[np.argmax(pbests_scores)]
    gbest_score = max(pbests_scores)

    history = [gbest_score]

    phi = c1 + c2
    k = 2 / abs(2 - phi - np.sqrt(phi**2 - 4 * phi))

    for _ in range(iterations):
        for i in range(num_particles):
            r1 = np.random.rand(num_courses)
            r2 = np.random.rand(num_courses)

            cognitive = c1 * r1 * (np.array(pbests[i]) - np.array(positions[i]))
            social = c2 * r2 * (np.array(gbest) - np.array(positions[i]))
            vel = k * w * (np.array(velocities[i]) + cognitive + social)
            pos = np.clip(np.array(positions[i]) + vel, 0, NUM_TIMESLOTS - 3)

            velocities[i] = vel.tolist()
            positions[i] = pos.tolist()

            schedule = build_schedule_from_position(courses, positions[i])

            if use_local_search:
                local_schedule = interchange(schedule)
                fitness_orig = fitness_with_constraints(schedule, prefs_teacher, prefs_class)
                fitness_local = fitness_with_constraints(local_schedule, prefs_teacher, prefs_class)

                if fitness_local > fitness_orig:
                    schedule = local_schedule
                    new_pos = []
                    for item in schedule:
                        start_ts = item[4]
                        new_pos.append(0 if start_ts == -1 else start_ts)
                    positions[i] = new_pos
                    fitness = fitness_local
                else:
                    fitness = fitness_orig
            else:
                fitness = fitness_with_constraints(schedule, prefs_teacher, prefs_class)

            if fitness > pbests_scores[i]:
                pbests[i] = positions[i].copy()
                pbests_scores[i] = fitness
                if fitness > gbest_score:
                    gbest = positions[i].copy()
                    gbest_score = fitness

        history.append(gbest_score)

    return gbest_score, history

iterations_sizes = [500, 1000, 2000, 3000, 4000, 5000, 6000]

results_pso1 = []
results_pso2 = []
results_pso3 = []
results_pso4 = []

for iterations in iterations_sizes:
    print(f"Rodando PSO c1=c2=2.05 w=0.8 com {iterations} iterações...")
    best_fitness, _ = run_course_timetabling_pso(2.05, 2.05, 0.8, num_particles=20, iterations=iterations, use_local_search=True)
    results_pso1.append(best_fitness)

    print(f"Rodando PSO c1=c2=2 w=0.8 com {iterations} iterações...")
    best_fitness, _ = run_course_timetabling_pso(2, 2, 0.8, num_particles=20, iterations=iterations, use_local_search=True)
    results_pso2.append(best_fitness)

    print(f"Rodando PSO c1=2.4 c2=2 w=0.9 com {iterations} iterações...")
    best_fitness, _ = run_course_timetabling_pso(2.4, 2, 0.9, num_particles=20, iterations=iterations, use_local_search=True)
    results_pso3.append(best_fitness)

    print(f"Rodando PSO c1=1.9 c2=2.1 w=0.6 com {iterations} iterações...")
    best_fitness, _ = run_course_timetabling_pso(1.9, 2.1, 0.6, num_particles=20, iterations=iterations, use_local_search=True)
    results_pso4.append(best_fitness)

# Gráfico
plt.figure(figsize=(10, 6))
plt.plot(iterations_sizes, results_pso1, marker='o', label='c1=c2=2.05, w=0.8')
plt.plot(iterations_sizes, results_pso2, marker='s', label='c1=c2=2, w=0.8')
plt.plot(iterations_sizes, results_pso3, marker='^', label='c1=2.4, c2=2, w=0.9')
plt.plot(iterations_sizes, results_pso4, marker='d', label='c1=1.9, c2=2.1, w=0.6')

plt.title("Comparação de Fitness por Iterações")
plt.xlabel("Iterações")
plt.ylabel("Melhor Fitness Final")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# results_pso = []
# results_spso = []

# for size in particle_sizes:
#     print(f"Rodando PSO tradicional com {size} partículas...")
#     best_fitness, _ = run_course_timetabling_pso(2.1, 1.9, 0.5, num_particles=size, iterations=1000, use_local_search=False)
#     results_pso.append(best_fitness)

#     print(f"Rodando SPSO (com busca local) com {size} partículas...")
#     best_fitness, _ = run_course_timetabling_pso(2.1, 1.9, 0.5, num_particles=size, iterations=1000, use_local_search=True)
#     results_spso.append(best_fitness)

# plt.figure(figsize=(10, 6))
# plt.plot(particle_sizes, results_pso, marker='o', label='PSO Tradicional (sem busca local)')
# plt.plot(particle_sizes, results_spso, marker='s', label='SPSO (com busca local)')
# plt.title("Comparação de Fitness por Tamanho de População")
# plt.xlabel("Número de Partículas")
# plt.ylabel("Melhor Fitness Final")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
