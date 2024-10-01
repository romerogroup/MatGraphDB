import torch


print(torch.__version__)
print(torch.cuda.is_available())


# from matgraphdb import DBManager


# if __name__=='__main__':
#     db=DBManager()
#     db.fix_chargemol_name()


# # with open(os.path.join(DB_DIR,'mp-1000.json')) as f:
# #     data = json.load(f)

# # print(data['chargemol_bonding_connections'])
# # print(data['chargemol_bonding_orders'])

# # data['chargemol_bonding_orders']=data['chargemol_bonding_connections']
# # data['chargemol_bonding_connections']=data['chargemol_bonding_orders']

# # with open(os.path.join(DB_DIR,'mp-1000.json'),'w') as f:
# #     json.dump(data, f, indent=4)