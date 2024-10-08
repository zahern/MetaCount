import logging
from statistics import mean
import matplotlib.pyplot as plt
import random

logger = logging.getLogger(__name__)


class Solution(dict):

    # Counter used to track solution progression
    sol_counter = 0

    def __init__(self, *arg, **kw):
        self['aic'] = 10000000.0
        self['bic'] = 10000000.0
        self['MAE'] = 10000000.0
        self['MSE'] = 10000000.0
        self['MAE_TEST'] = 10000000.0
        self['MSE_TEST'] = 10000000.0
        self['MAE_VAL'] = 10000000.0
        self['MSE_VAL'] = 10000000.0
            #self['MAPE'] = MAE.get('MAPE')
        self['RMSE_VAL'] = 10000000.0
        self['RMSE_TEST'] = 10000000.0
        self['RMSE'] = 10000000.0
        self['layout'] = None
        self['sol_num'] = Solution.sol_counter
        self['num_parm'] = 1000
        self['pval_exceed'] = 100
        self['pval_percentage'] = 100
        self['loglik'] = -1000000000
        self['simple'] = 'f'
        self['fixed_fit'] = None
        self['rdm_fit'] = None
        self['rdm_cor_fit'] = None
        self['zi_fit'] = None
        self['pvalues'] = None
        Solution.sol_counter += 1
        super(Solution, self).__init__(*arg, **kw)

    def add_objective(self, bic=None, TRAIN=None, loglik=None, num_parm=None, pval_exceed=None, aic=None, GOF = None, TEST = None, VAL = None):
        if bic is not None:
            
            self['bic'] = bic
        if TEST is not None:
            self['MAE_TEST'] = TEST.get('MAE')
            self['MSE_TEST'] = TEST.get('MSE')
            #self['MAPE'] = MAE.get('MAPE')
            self['RMSE_TEST'] = TEST.get('RMSE')
        
        if VAL is not None:
            self['MAE_VAL'] = VAL.get('MAE')
            self['MSE_VAL'] = VAL.get('MSE')
            #self['MAPE'] = MAE.get('MAPE')
            self['RMSE_VAL'] = VAL.get('RMSE')
        
        if TRAIN is not None:
        
            self['MAE'] = TRAIN.get('MAE')
            self['MSE'] = TRAIN.get('MSE')
            #self['MAPE'] = MAE.get('MAPE')
            self['RMSE'] = TRAIN.get('RMSE')
           
        if loglik is not None:
            self['loglik'] = loglik
        if aic is not None:

            self['aic'] = aic
        if num_parm is not None:
            self['num_parm'] = num_parm
        if pval_exceed is not None:
            self['pval_exceed'] = pval_exceed
            self['pval_percentage'] = float(pval_exceed/len(self['pvalues']))
        if GOF is not None:
            for k in GOF.keys():
                self[k] = GOF[k]
                #self.__setattr__(k, GOF[k])
                

    def add_layout(self, layout):
        self['layout'] = layout

    def add_names(self, fixed_fit, rdm_fit, rdm_cor_fit, simple=0, zi_fit=None, pvalues=None):
        self['fixed_fit'] = fixed_fit
        self['rdm_fit'] = rdm_fit
        self['rdm_cor_fit'] = rdm_cor_fit
        self['simple'] = simple
        self['zi_fit'] = zi_fit
        self['pvalues'] = pvalues


class Pareto(object):
    
    def __init__(self, obj_key1='bic', obj_key_2='MAE', multiobjective=True):
        self.obj_key_1 = obj_key1
        self.obj_key_2 = obj_key_2
        self.multi_objective = multiobjective
        self.minimise_1 = 1
        self.minimise_2 = 1
        self.did_change = 1
        self.Structs = []
        self.Pareto_F = []
        self.mean_ob1 = None  # store the mean of the objective functions
        self.mean_ob2 = None

    def get_objective_is_multi(self):
        return self.multi_objective

    def add_Structs(self, Struct):
        self.Structs.append(Struct)

    def did_it_change(self):
        if self.did_change:
            self.did_change = 0
            return 1
        else:
            return 0

    def run(self, add_Struct=None):
        size_of = len(self.Pareto_F)
        if add_Struct is not None:

            self.add_Structs(add_Struct)
        sol_stuff = self.non_dominant_sorting(self.Structs)

        Fronts = self.get_fronts(sol_stuff)

        Pareto_FRONT = self.pareto(Fronts, sol_stuff)
        self.Structs = Pareto_FRONT
        self.Pareto_F = Pareto_FRONT
        if self.multi_objective:
            self.update_means()
        if size_of != len(self.Pareto_F):
            self.did_change = 1
            return 1
        else:
            return 0

    def pareto_run(self, Struct, SAVE=False, append_new=None):
        if append_new is not None:
            Struct.insert(0, append_new)
        sol_stuff = self.non_dominant_sorting(Struct)

        Fronts = self.get_fronts(sol_stuff)

        Struct = self.pareto(Fronts, sol_stuff)
        if SAVE:
            self.Structs = Struct
            self.Pareto_F = Struct
        return Struct

    def evaluate_frontier_against_new_sln(self, Struct):
        'Function to check if a new solution dominates another in the existing set'

        val_key_1 = self.obj_key_1
        val_key_2 = self.obj_key_2

        for j in range(len(self.Pareto_F)):

            if self.minimise_2:
                dominance = self.check_dominance([Struct[val_key_1], Struct[val_key_2]],
                                                 [self.Pareto_F[j][val_key_1], self.Pareto_F[j][val_key_2]])
                if dominance:

                    self.add_Structs(Struct)
                    self.run()
                    return
                    # self.Pareto_F.append(Struct)

            else:  # minimize the second objective
                dominance = self.check_dominance([-Struct[val_key_1], -Struct[val_key_2]],
                                                 [-self.Pareto_F[j][val_key_1], -self.Pareto_F[j][val_key_2]])
                if dominance:
                    self.add_Structs(Struct)
                    self.run()
                    return

    
    
    def is_pareto_efficient(self, new_dict, dict_list):
        if new_dict in dict_list:
            return True
        for dictionary in dict_list:
            if dictionary[self.obj_key_1] <= new_dict[self.obj_key_1] and dictionary[self.obj_key_2] <= new_dict[self.obj_key_2]:
                if dictionary[self.obj_key_1] < new_dict[self.obj_key_1] or dictionary[self.obj_key_2] < new_dict[self.obj_key_2]:
                    return False
        return True




    
    
    def check_if_dominance(self, Structs, New_Struct, return_struct = 0):
        """
        Funtion for non-dominant for newly proposed sln
        ni - the number of solutions which dominate the solution i
        si - a set of solutions which the solution i dominates

        Inputs: List containing set of solutions

        Output: return false if does not dominate, otherwise add to the pareto set

        """

        val_key_1 = self.obj_key_1
        val_key_2 = self.obj_key_2
        dominance = False
        if len(Structs) == 0:
            Structs.append(New_Struct)
            Structs = self.pareto_run(Structs, True)
            if return_struct:
                return True, Structs
            else:
                return True

        if isinstance(Structs, list):

            for j in range(len(Structs)):

                if self.minimise_1 and self.minimise_2:  # minimise both
                    dominance = self.check_dominance([New_Struct[val_key_1], New_Struct[val_key_2]],
                                                     [Structs[j][val_key_1], Structs[j][val_key_2]])
                    if dominance:

                        Structs.append(New_Struct)
                        Structs = self.pareto_run(Structs, True)
                        print('new solution, pareto optimal')
                        if return_struct:
                            return dominance, Structs
                        else:
                            return dominance  
                        break
                elif self.minimise_1 and not self.minimise_2:  # minimise 1 and maximise 2
                    dominance = self.check_dominance([New_Struct[val_key_1], -New_Struct[val_key_2]],
                                                     [Structs[j][val_key_1], -Structs[j][val_key_2]])
                    if dominance:

                        Structs.append(New_Struct)
                        Structs = self.pareto_run(Structs, True)
                        print('new solution, pareto optimal')
                        if return_struct:
                            return dominance, Structs
                        else:
                            return dominance  
                        break
                elif not self.minimise_1 and self.minimise_2:  # maximise 1 and minise 2
                    dominance = self.check_dominance([-New_Struct[val_key_1], New_Struct[val_key_2]],
                                                     [-Structs[j][val_key_1], Structs[j][val_key_2]])
                    if dominance:

                        Structs.append(New_Struct)
                        Structs = self.pareto_run(Structs, True)
                        print('new solution, pareto optimal')
                        if return_struct:
                            return dominance, Structs
                        else:
                            return dominance  
                        break

                else:  # maximise both
                    dominance = self.check_dominance([-New_Struct[val_key_1], -New_Struct[val_key_2]],
                                                     [-Structs[j][val_key_1], -Structs[j][val_key_2]])
                    if dominance:

                        Structs.append(New_Struct)
                        Structs = self.pareto_run(Structs, True)
                        print('new solution, pareto optimal')
                        if return_struct:
                            return dominance, Structs
                        else:
                            return dominance   
                        break
            if return_struct:
                return dominance, Structs
            else:
                return dominance        
          
        else:
            print('not a list')
            if self.minimise_2:
                dominance = self.check_dominance([New_Struct[val_key_1], New_Struct[val_key_2]],
                                                 [Structs[val_key_1], Structs[val_key_2]])
                if dominance:

                    Structs.append(New_Struct)
                    Structs = self.pareto_run(Structs)
                    print('new solution, pareto optimal')

                else:
                    dominance = self.check_dominance([Structs[val_key_1], Structs[val_key_2]],
                                                     [New_Struct[val_key_1], New_Struct[val_key_2]])
                    if dominance:
                        Structs.append(New_Struct)
                        Structs = self.pareto_run(Structs)
                        print('new solution, pareto optimal')

            else:  # minimize the second objective
                dominance = self.check_dominance([-New_Struct[val_key_1], -New_Struct[val_key_2]],
                                                 [-Structs[val_key_1], -Structs[val_key_2]])
                if dominance:
                    Structs.append(New_Struct)
                    Structs = self.pareto_run(Structs)
                    print('new solution, pareto optimal')

                else:
                    dominance = self.check_dominance([-Structs[val_key_1], -Structs[val_key_2]],
                                                     [-New_Struct[val_key_1], -New_Struct[val_key_2]])
                    if dominance:
                        Structs.append(New_Struct)
                        Structs = self.pareto_run(Structs)
                        print('new solution, pareto optimal')
            if return_struct:
                return dominance, Structs
            else:
                return dominance      

    def get_fronts(self, Struct) -> dict:  # TODO get rid of HM
        """
        Funtion for non-dominant sorting of the given set of solutions
        ni - the number of solutions which dominate the solution i
        si - a set of solutions which the solution i dominates

        Inputs: List containing set of solutions

        Output: Dict with keys indicating the Pareto rank and values containing indices of solutions in Input
        """
        si = {}
        ni = {}
        val_key_1 = self.obj_key_1
        val_key_2 = self.obj_key_2

        if len(Struct) == 1:
            # Identify solutions in each front
            Fronts = {}
            itr = 0

            Fronts.update({'F_{}'.format(itr): [0]})

            logger.info("Fronts: {}".format(str(Fronts)))
            return Fronts

        for i in range(len(Struct)):
            sp_i = []
            np_i = 0
            for j in range(len(Struct)):
                if i != j:
                    if self.minimise_2:
                        dominance = self.check_dominance([Struct[i][val_key_1], Struct[i][val_key_2]],
                                                         [Struct[j][val_key_1], Struct[j][val_key_2]])
                        if dominance:
                            sp_i.append(j)
                        else:
                            dominance = self.check_dominance([Struct[j][val_key_1], Struct[j][val_key_2]],
                                                             [Struct[i][val_key_1], Struct[i][val_key_2]])
                            if dominance:
                                np_i += 1
                    else:  # minimize the second objective
                        dominance = self.check_dominance([-Struct[i][val_key_1], -Struct[i][val_key_2]],
                                                         [-Struct[j][val_key_1], -Struct[j][val_key_2]])
                        if dominance:
                            sp_i.append(j)
                        else:
                            dominance = self.check_dominance([-Struct[j][val_key_1], -Struct[j][val_key_2]],
                                                             [-Struct[i][val_key_1], -Struct[i][val_key_2]])
                            if dominance:
                                np_i += 1

            si.update({i: sp_i})
            ni.update({i: np_i})
        # Identify solutions in each front
        Fronts = {}
        itr = 0
        for k in range(len(ni)):
            Fi_idx = [key for key, val in ni.items() if val == k]
            if len(Fi_idx) > 0:
                Fronts.update({'F_{}'.format(itr): Fi_idx})
                itr += 1

        logger.info("Fronts: {}".format(str(Fronts)))
        return Fronts

    def weighted_obj(self, sln):
        val = sln[self.obj_key_1]/self.mean_ob1 + \
            sln[self.obj_key_2]/self.mean_ob2
        return val

    def calculate_difference(self, SLN_1, SLN_2):
        'calulates the difference betwene two solutions'
        val = self.delta_calc(SLN_1, SLN_2)
        if val < 0:
            return 1
        else:
            return 0

    def check_dominance(self, obj1, obj2):
        """
        Function checks dominance between solutions for two objective functions
        Inputs: obj1 - List containing values of the two objective functions for solution 1
                obj2 - List containing values of the two objective functions for solution 2
        Output: Returns True if solution 1 dominates 2, False otherwise
        """
        if len(obj1) != 2:
            desired_keys = [self.obj_key_1, self.obj_key_2]
            # subdictionaries
            ob1 = {k: obj1[k] for k in obj1.keys() if k in desired_keys}
            ob2 = {k: obj2[k] for k in obj2.keys() if k in desired_keys}
        else:
            ob1 = obj1
            ob2 = obj2
        indicator = False
        for a, b in zip(ob1, ob2):
            if a < b:
                indicator = True
            # if one of the objectives is dominated, then return False
            elif a > b:
                return False

        return indicator

    def update_means(self):
        f1 = mean(self.Pareto_F[x][self.obj_key_1]
                  for x in range(len(self.Pareto_F)))
        f2 = mean(self.Pareto_F[x][self.obj_key_2]
                  for x in range(len(self.Pareto_F)))
        self.mean_ob1 = f1
        self.mean_ob2 = f2

    def pareto(self, Fronts, Struct) -> list:
        size = len(self.Pareto_F)
        if Fronts is None:
            Fronts = self.get_fronts(Struct)

        if len(Struct) == 1:
            self.Pareto_F = Struct
            return Struct

        Pareto_front_id = []
        for k, v in Fronts.items():
            if len(v) > 0:
                Pareto_front_id = Fronts.get(k)
                break
        Pareto_front = [Struct[x] for x in Pareto_front_id]
        self.Pareto_F = Pareto_front
        if size != len(self.Pareto_F):
            self.update_means()
        return Pareto_front

    def non_dominant_sorting(self, Struct):
        Front = self.get_fronts(Struct)
        crowd = self.crowding_dist(Front, Struct)
        Sorted_Slns = self.sort_Slns(Front, crowd, Struct)
        return Sorted_Slns

    def delta_calc(self, Proposed_Struct, Current_Struct):
        delta = (Proposed_Struct[self.obj_key_1] - Current_Struct[self.obj_key_1])/self.mean_ob1\
            + (Proposed_Struct[self.obj_key_2] -
               Current_Struct[self.obj_key_2])/self.mean_ob2

        return delta

    def find_bestsol(self, Struct):
        f1 = mean(Struct[x][self.obj_key_1] for x in range(len(Struct)))
        f2 = mean(Struct[x][self.obj_key_2] for x in range(len(Struct)))
        self.mean_ob1 = f1
        self.mean_ob2 = f2
        f_joined = f1+f2
        w_1 = f1/(f1+f2)
        w_2 = f2/(f1+f2)

        max_obj1 = max(Struct[x][self.obj_key_1] for x in range(len(Struct)))
        min_obj1 = min(Struct[x][self.obj_key_1] for x in range(len(Struct)))
        weights_obj1 = [(Struct[x][self.obj_key_1])-min_obj1 /
                        (max_obj1-min_obj1) for x in range(len(Struct))]

        if self.multi_objective:
            max_obj2 = max(Struct[x][self.obj_key_2]
                           for x in range(len(Struct)))
            min_obj2 = min(Struct[x][self.obj_key_2]
                           for x in range(len(Struct)))
            weights_obj2 = [(Struct[x][self.obj_key_2])-min_obj2 /
                            (max_obj2-min_obj2) for x in range(len(Struct))]

            weights = [weights_obj1[x] + weights_obj2[x]
                       for x in range(len(Struct))]

            weights_alt = [f_joined*(w_2*Struct[x][self.obj_key_1] +
                                     w_1*Struct[x][self.obj_key_2]) for x in range(len(Struct))]
            best_solid = weights.index(min(weights))
            best_solid = weights_alt.index(min(weights_alt))
        else:
            weights = weights_obj1
            best_solid = weights.index(min(weights))

        logger.info("best sol for local search: {}".format(Struct[best_solid]))
        return Struct[best_solid]

    def sort_Slns(self, Fronts, v_dis, Struct):
        """
        Function to sort memory from best solution to worst solution
        Inputs:
        Fronts-Dict with keys indicating Pareto rank and values indicating indices of solutions belonging to the rank
        v_dis - Dict with keys indicating index of solution in memory and value indicating crowding distance
        Output:
        Sorted_HM - Sorted list of solutions
        """
        Sorted_Sln_id = []
        for k, v in Fronts.items():
            pareto_sols = {key: val for key, val in v_dis.items()
                           if key in Fronts.get(k)}
            Sorted_Sln_id.extend([ke for ke, va in
                                 sorted(pareto_sols.items(),
                                        key=lambda item: item[1])])

        Sorted_Slns = [Struct[x] for x in Sorted_Sln_id]
        if len(Struct) != 2 and len(Sorted_Sln_id) != 6:
            if len(Sorted_Sln_id) != len(Struct):
                print('error potentianlly?')
        return Sorted_Slns

    def crowding_dist(self, Fronts, Struct):
        """
        Function to estimate crowding distance between 2 solutions
        Inputs:
        Fronts-Dict with keys indicating Pareto rank and values indicating indices of solutions belonging to the rank
        HM - List of solutions
        """
        v_dis = {}
        val_key_1 = self.obj_key_1
        val_key_2 = self.obj_key_2

        for v in Fronts.values():
            v.sort(key=lambda x: Struct[x][val_key_1])
            for i in v:
                v_dis.update({i: 0})
        # Calculate crowding distance based on first objective
        for v in Fronts.values():
            for j in v:
                if v[0] == j or v[-1] == j:
                    v_dis.update({j: 1000000})
                else:
                    dis = abs(v_dis.get(j) +
                              ((Struct[v[v.index(j) + 1]][val_key_1] -
                                Struct[j][val_key_1]) / (max(Struct[x][val_key_1] for x in
                                                             range(len(Struct))) -
                                                         min(Struct[x][val_key_1] for x in
                                                             range(len(Struct))))))
                    v_dis.update({j: dis})

        # Calculate crowding distance based on second objective
        q_dis = {}
        for v in Fronts.values():
            v.sort(key=lambda x: Struct[x][val_key_2])
            for k in v:
                q_dis.update({k: 0})
        for v in Fronts.values():
            for k in v:
                if v[0] == k or v[-1] == k:
                    q_dis.update({k: 1000000})
                else:
                    dis = abs(q_dis.get(k) + ((Struct[v[v.index(k)+1]][val_key_2] -
                                               Struct[k][val_key_2])
                                              / (max(Struct[x][val_key_2] for x
                                                     in range(len(Struct))) -
                                                  min(Struct[x][val_key_2] for x in
                                                      range(len(Struct))))))
                    q_dis.update({k: dis})
        # Adding crowding distance from both objectives
        crowd = {k: q_dis[k] + v_dis[k] for k in v_dis.keys()}
        return crowd

_pareto = Pareto()
_solution = Solution()
# p = Pareto('bic', 'MAE')
# sol_stuff = list()
# #test code

# for i in range(1000):
#     bic = random.randint(0, 1000)
#     MAE = random.randint(0, 1000)
#     sol_stuff.append(Solution(bic=bic, MAE = MAE))

# #main driver
# print(len(sol_stuff))
# plt.scatter([x['bic'] for x in sol_stuff], [x['MAE'] for x in sol_stuff])
# print('the first instance of the solutions')
# sol_stuff = p.non_dominant_sorting(sol_stuff)
# print(len(sol_stuff))
# Fronts = p.get_fronts(sol_stuff)
# print(Fronts)
# Pareto_FRONT = p.pareto(Fronts, sol_stuff)
# print(Pareto_FRONT)
# print(2)


# plt.scatter([x['bic'] for x in Pareto_FRONT], [x['MAE'] for x in Pareto_FRONT])
# plt.show()
