import json
import os
import random
from typing import Any, List
import numpy as np
import pandas as pd
from dataclasses import asdict, dataclass
from transformermath.tokenizer import AdditionTokenizer, SubtractionTokenizer, SequencesTokenizer
import copy


# A + B = C
@dataclass
class ArithmeticItem:
    A: int
    B: int
    C: int


class DataCreator:

    def __init__(self) -> None:
        self.DATA_COLUMN_NAME = "data"
        self.DATA_LABEL_NAME = "labels"
        self.DATA_INDEX_NAME = "data_id_mappings"
        self.END_TOKEN = str("E")
        self.PAD_TOKEN = str("P")
    
    def two_number_sequence_A_B(self,digits):
        increment = random.randint(1, 9)
        min_value, max_value = 10 ** (digits - 1), (10 ** digits) - 1
        A = random.randint(min_value, max_value)
        B = A + increment
        C = B + increment
        return A,B,C

    def randA_specifyB(self, digits, addend):
        min_value, max_value = 10 ** (digits - 1), (10 ** digits) - 1
        A = random.randint(min_value, max_value)
        B = addend
        C = A + B
        return A,B,C
    
    def specifyA_randB(self, digits, addend):
        min_value, max_value = 10 ** (digits - 1), (10 ** digits) - 1
        A = addend
        B = random.randint(min_value, max_value)
        C = A + B
        return A,B,C
    
    def randA_randB(self, digits):
        min_value, max_value = 10 ** (digits - 1), (10 ** digits) - 1
        A = random.randint(min_value, max_value)
        B = random.randint(min_value, max_value)
        C = A + B
        return A,B,C
    
    # ! subtraction
    def randA_randB_for_subtraction(self, digits_A=5, digits_B=3):
        min_value_A, max_value_A = 10 ** (digits_A - 1), (10 ** digits_A) - 1
        min_value_B, max_value_B = 10 ** (digits_B - 1), (10 ** digits_B) - 1
        A = random.randint(min_value_A, max_value_A)
        B = random.randint(min_value_B, max_value_B)
        C = A - B  
        return A, B, C
    
    """
    A: e.g. {0,1,2,3,4}
    B: e.g. {5,6,7,8,9}
    """
    def setA_setB(self, digits: int, setA: List[int], setB: List[int]):
        A_list = []
        B_list = []
        
        newSetA = copy.deepcopy(setA)
        newSetB = copy.deepcopy(setB)
        if 0 in newSetA:
            newSetA.remove(0)
        if 0 in newSetB:
            newSetB.remove(0)
        
        # no zero for first digit
        A_list.append(str(random.choice(newSetA)))
        B_list.append(str(random.choice(newSetB)))
        
        # rest of digits
        for _ in range(1, digits):
            A_list.append(str(random.choice(setA)))
            B_list.append(str(random.choice(setB)))
        
        A = int(''.join(A_list))
        B = int(''.join(B_list))
        C = A + B
        
        return A, B, C
    
    def generate_sequence_items(self, digits, equations, exclude_data):
        
        equation_index = 0
        data_items = list()

        while equation_index < equations:
            A,B,C = self.two_number_sequence_A_B(digits=digits)
            data_item = ArithmeticItem(A=A,B=B,C=C)

            # re generate if data item has been generated before
            if data_item in data_items:
                continue

            # re generate numbers if equations are in the excluded data
            if len(exclude_data) != 0 and data_item in exclude_data:
                continue

            data_items.append(data_item)
            equation_index += 1

            if equation_index % 100 == 0:
                print(f"|Equations Generated: {equation_index}/{equations}|")

        return data_items
    
    """
    Formats data into next token prediction
    """
    def construct_next_token_data(self, data_items: List[ArithmeticItem], operation: str, max_seq_len = None, equals: str = "="):
        OPERATION = str(operation)
        EQUALS = str(equals)
        
        data, labels, ids = [], [], []
        for equation_index, item in enumerate(data_items):
            A,B,C = item.A, item.B, item.C
            
            answer_arr = [str(digit) for digit in str(C)] + [self.END_TOKEN] 

            # e.g. [4 2 1 + 2 5 3 =]
            initial_subinstance = [str(digit) for digit in str(A)] + [OPERATION] + [str(digit) for digit in str(B)] + [EQUALS] 
            
            # e.g. [[4 2 1 + 2 5 3 =], ... []] 
            sub_instances = []

            # create the sub instance set
            for i, _ in enumerate(answer_arr):
                if i == 0:
                    sub_instances.append(initial_subinstance)
                    continue
                prev_answer = answer_arr[i-1]
                x = sub_instances[i-1] + [prev_answer]
                sub_instances.append(x)

            # validation test
            if len(sub_instances) != len(answer_arr):
                raise ValueError("labels not generated correctly.")
            
            # add to overall dataset
            for i in range(len(sub_instances)):
                data.append(sub_instances[i])
                labels.append(answer_arr[i])
                ids.append(equation_index)
            
        # pad input data
        if max_seq_len:
            max_length = max_seq_len
        else:
            max_length = max(len(item) for item in data)  
        padded_data = np.array([item + [self.PAD_TOKEN] * (max_length - len(item)) for item in data])
        
        if len(data) == len(labels) == len(ids): 
            return list(padded_data), labels, ids
        else:
            raise ValueError("Data Length != Label Length != Ids Length")
        
    
    def generate_addition_items(self, digits, equations, exclude_data):
        
        equation_index = 0
        data_items = list()

        while equation_index < equations:
            A,B,C = self.randA_randB(digits=digits)
            data_item = ArithmeticItem(A=A,B=B,C=C)

            # re generate if data item has been generated before
            if data_item in data_items:
                continue

            # re generate numbers if equations are in the excluded data
            if len(exclude_data) != 0 and data_item in exclude_data:
                continue

            data_items.append(data_item)
            equation_index += 1

            if equation_index % 100 == 0:
                print(f"|Equations Generated: {equation_index}/{equations}|")

        return data_items

    def generate_subtraction_items(self, equations, data_type_func, digitA = None,
                                   digitB = None, exclude_data = None, setA = None, setB = None):
        
        equation_index = 0
        data_items = list()

        while equation_index < equations:
            
            if digitA and digitB:
                A,B,C = data_type_func(digitA,digitB)
            else:
                A,B,C = data_type_func()

            data_item = ArithmeticItem(A=A,B=B,C=C)

            # re generate if data item has been generated before
            if data_item in data_items:
                continue

            # re generate numbers if equations are in the excluded data
            if exclude_data and data_item in exclude_data:
                continue

            data_items.append(data_item)
            equation_index += 1

            if equation_index % 1000 == 0:
                print(f"Dataset Size: ({equation_index}/{equations})")
        return data_items

    def create_encoded_dataframe(self, input_data: List[Any], targets: List[Any], data_mappings: List[int], replacer) -> pd.DataFrame:
        REPLACER = replacer
        # construct dataframe
        data = pd.DataFrame({
            self.DATA_COLUMN_NAME: input_data,
            self.DATA_LABEL_NAME: targets,
            self.DATA_INDEX_NAME: data_mappings,
        })

        # ENCODE tokens to numbers
        encoded_df = data.copy(deep=True)
        encoded_df[self.DATA_COLUMN_NAME] = encoded_df[self.DATA_COLUMN_NAME].apply(lambda seq: [REPLACER.get(str(item), item) for item in seq])
        encoded_df[self.DATA_LABEL_NAME] = encoded_df[self.DATA_LABEL_NAME].apply(lambda label: REPLACER.get(str(label), label))

        return encoded_df


    def save_item(self, items, file_name):
        items_dict = [asdict(item) for item in items]
        with open(file_name, 'w') as f:
            json.dump(items_dict, f, indent=4)
    
    def data_csv_save(self, dataframe, file_name):
        data_directory = 'data'
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        full_file_path = os.path.join(data_directory, file_name)
        dataframe.to_csv(full_file_path, index=False)

    def save_dataframe(self, dataframe, file_name):
        data_directory = 'data'
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        full_file_path = os.path.join(data_directory, file_name)
        dataframe.to_csv(full_file_path, index=False)

    def generate_addition_data(self, num_equations: int, n_digits: int, 
                               exclude_items: List[ArithmeticItem] = [], 
                               maximum_sequence_length = None):

        items = self.generate_addition_items(digits=n_digits, equations=num_equations,
                                                                exclude_data=exclude_items)
        
        X, Y, Ids = self.construct_next_token_data(data_items=items, operation=str("+"), max_seq_len = maximum_sequence_length)
        data = self.create_encoded_dataframe(input_data=X, targets=Y, 
                                                           data_mappings=Ids, replacer=AdditionTokenizer().REPLACER)

        Xtokenized, Ytokenized = data["data"], data["labels"]

        return list(Xtokenized), list(Ytokenized), items, data
        #return train_data, train_items, trainX, trainY
    
    def generate_subtraction_data(self, num_equations: int, digitsA: int, digitsB: int,
                              exclude_items: List[ArithmeticItem] = [], 
                              maximum_sequence_length = None):

        items = self.generate_subtraction_items(equations=num_equations,
                                                    data_type_func=self.randA_randB_for_subtraction,
                                                    digitA=digitsA, digitB=digitsB,
                                                    exclude_data=exclude_items)

        X, Y, Ids = self.construct_next_token_data(data_items=items, 
                                                max_seq_len=maximum_sequence_length,
                                                operation=str("-"))
        data = self.create_encoded_dataframe(input_data=X, targets=Y, 
                                                data_mappings=Ids, replacer=SubtractionTokenizer().REPLACER)

        Xtokenized, Ytokenized = data["data"], data["labels"]
        return list(Xtokenized), list(Ytokenized), items, data

    def generate_sequence_data(self, num_equations: int, n_digits: int, 
                               exclude_items: List[ArithmeticItem] = [], 
                               maximum_sequence_length = None):

        items = self.generate_sequence_items(digits=n_digits, equations=num_equations,
                                                                exclude_data=exclude_items)
        
        X, Y, Ids = self.construct_next_token_data(data_items=items, operation="<", equals=">", max_seq_len=maximum_sequence_length)
        data = self.create_encoded_dataframe(input_data=X, targets=Y, 
                                            data_mappings=Ids, replacer=SequencesTokenizer().REPLACER)

        Xtokenized, Ytokenized = data["data"], data["labels"]

        return list(Xtokenized), list(Ytokenized), items, data
        

    