# -*- coding: utf-8 -*-
requirement_weight = [
    {'普通护照签发、换发、补发及加注-首次申请护照': 100, '首次申请护照': 100, '首次': 50, '护照': 50, '外国': 10, '定居': 10, '探亲': 10, '学习': 10,
     '就业': 10, '旅行': 10, '商务活动': 10,
     '非公务原因': 10, '出国': 10, '申请': 10, '公民': 10},
    {'普通护照签发、换发、补发及加注-护照加注': 100, '护照加注': 100, '曾用名': 30, '曾用名加注': 50, '繁体': 30, '汉字': 30, '英文': 30, '多音字': 30,
     '非标准': 30, '汉语': 30, '发音': 30,
     '姓名': 30, '姓名加注': 50, '曾持照': 30, '加注项': 50, '加注': 50, '曾持照加注': 50, '护照': 50, '外国': 10, '定居': 10, '探亲': 10,
     '学习': 10, '就业': 10, '旅行': 10, '商务活动': 10, '非公务原因': 10, '出国': 10, '申请': 10, '公民': 10},
    {'普通护照签发、换发、补发及加注-护照补发': 100, '护照补发': 100, '损毁': 60, '不能': 10, '使用': 10, '被盗': 60, '遗失': 60, '补发': 60, '护照': 50,
     '外国': 10, '定居': 10, '探亲': 10,
     '学习': 10, '就业': 10, '旅行': 10, '商务活动': 10, '非公务原因': 10, '出国': 10, '申请': 10, '公民': 10},
    {'普通护照签发、换发、补发及加注-换发护照': 100, '换发护照': 100, '有效期': 30, '届满': 30, '签证页': 30, '使用': 10, '完毕': 10, '换发': 50, '护照': 50,
     '外国': 10, '定居': 10, '探亲': 10, '延期': 30, '更换': 40,
     '学习': 10, '就业': 10, '旅行': 10, '商务活动': 10, '非公务原因': 10, '出国': 10, '申请': 10, '公民': 10},
    {'机动车登记与临时通行牌证核发-机动车登记': 100, '初次申领机动车号牌和行驶证': 100, '汽车': 30, '机动车': 30, '初次': 30, '申领': 10, '号牌': 50, '行驶证': 30,
     '车辆管理所': 10, '登记': 10,
     '住所地': 10},
    {'机动车登记与临时通行牌证核发-机动车登记': 100, '改变车身颜色': 100, '汽车': 30, '机动车': 30, '改变': 10, '车身': 30, '颜色': 50, '申请': 10, '变更': 30,
     '登记': 10, '登记地': 10,
     '车辆管理所': 10},
    {'机动车登记与临时通行牌证核发-机动车登记': 100, '更换发动机': 100, '汽车': 30, '机动车': 30, '更换': 50, '发动机': 50, '申请': 10, '变更': 30, '登记': 10,
     '登记地': 10, '车辆管理所': 10},
    {'机动车登记与临时通行牌证核发-机动车登记': 100, '更换车身车架': 100, '汽车': 30, '机动车': 30, '更换': 50, '车身': 50, '车架': 50, '申请': 10, '变更': 30,
     '登记': 10, '登记地': 10,
     '车辆管理所': 10},
    {'机动车登记与临时通行牌证核发-机动车登记': 100, '因质量问题更换整车': 100, '汽车': 30, '机动车': 30, '更换': 50, '质量': 30, '问题': 10, '整车': 50,
     '申请': 10, '变更': 30, '登记': 10,
     '登记地': 10, '车辆管理所': 10},
    {'机动车登记与临时通行牌证核发-机动车登记': 100, '改变运营模式和使用性质': 100, '汽车': 30, '机动车': 30, '改变': 30, '营运': 50, '非营运': 50, '模式': 10,
     '使用': 10, '性质': 10, '申请': 10,
     '变更': 30, '登记': 10, '登记地': 10, '车辆管理所': 10},
    {'机动车登记与临时通行牌证核发-机动车登记': 100, '车辆管理所管辖区域改变': 100, '汽车': 30, '机动车': 30, '车辆管理所': 50, '改变': 30, '住所': 50, '迁入': 50,
     '迁出': 50, '管辖': 50, '区域': 10,
     '所有人': 30},
    {'机动车登记与临时通行牌证核发-机动车登记': 100, '已注册登记车辆所有权发生改变': 100, '汽车': 30, '机动车': 30, '注册': 10, '登记': 10, '所有权': 50, '转移': 50,
     '迁出': 50, '登记地': 10,
     '车辆管理所': 10,
     '申请': 10},
    {'机动车登记与临时通行牌证核发-机动车登记': 100, '机动车抵押': 100, '机动车': 30, '汽车': 30, '所有人': 10, '消灭': 10, '解除': 10, '抵押物': 50,
     '抵押': 50, '抵押权': 50, '登记': 10,
     '登记地': 10, '车辆管理所': 10, '申请': 10},
    {'机动车登记与临时通行牌证核发-机动车登记': 100, '已达到国家强制报废标准': 100, '汽车': 30, '机动车': 30, '回收企业': 30, '交售': 10, '解体': 10, '回收证明': 30,
     '达到': 10, '强制': 50, '报废': 50,
     '标准': 50, '车辆管理所': 10},
    {'机动车登记与临时通行牌证核发-机动车登记': 100, '灭失': 100, '汽车': 30, '机动车': 30, '所有人': 10, '申请': 10, '注销': 10, '登记': 10, '登记地': 10,
     '车辆管理所': 10},
    {'机动车登记与临时通行牌证核发-机动车登记': 100, '因故不在我国境内使用': 100, '汽车': 30, '机动车': 30, '因故': 10, '不在': 10, '境内': 50, '使用': 30,
     '所有人': 10, '申请': 10, '我国': 30,
     '注销': 10, '登记': 10, '登记地': 10, '车辆管理所': 10},
    {'机动车登记与临时通行牌证核发-机动车登记': 100, '因质量问题退车': 100, '汽车': 30, '机动车': 30, '质量': 50, '问题': 50, '退车': 50, '所有人': 10,
     '申请': 10, '注销': 10, '登记': 10,
     '登记地': 10, '车辆管理所': 10},
    {'机动车登记与临时通行牌证核发-机动车登记': 100, '机动车登记被依法撤销': 100, '汽车': 30, '机动车': 30, '登记': 30, '依法': 30, '撤销': 50, '登记地': 10,
     '车辆管理所': 10, '注销': 10},
    {'机动车登记与临时通行牌证核发-机动车登记': 100, '被依法收缴并强制报废的': 100, '汽车': 30, '机动车': 30, '达到': 10, '国家': 10, '强制': 10, '报废': 10,
     '标准': 10, '依法': 30, '收缴': 50,
     '登记地': 10, '车辆管理所': 10, '注销': 10, '登记': 10},
    {'客运出租汽车经营资格证件核发-客运出租汽车经营资格证核发': 100, '客运出租汽车经营资格证核发': 100, '核发': 10, '客运': 30, '出租': 80, '汽车': 30, '经营': 30,
     '资格证': 30, '资格': 10, '车': 10},
    {'客运出租汽车经营资格证件核发-网络预约出租汽车驾驶员证核发-网络预约出租汽车驾驶员证核发': 100, '网络预约出租汽车驾驶员证核发': 100, '网络预约出租汽车': 80, '网络': 40, '预约': 80,
     '出租': 50, '汽车': 40, '驾驶员证': 30, '核发': 10, '申请': 10},
    {'客运出租汽车经营资格证件核发-驾驶员客运资格证核发': 100, '驾驶员客运资格证核发': 100, '客运': 50, '驾驶员': 30, '资格证': 30, '核发': 10, '职业培训': 30,
     '考试合格': 10, '初中': 10, '健康': 10},
    {'教师资格的认定': 100, '认定教师资格': 100, '认定': 10, '学校': 30, '教师': 80, '资格': 30, '考试合格': 10, '天津': 10, '学历': 10, '普通话': 10},
    {'养犬登记': 100, '养犬': 80, '饲养': 50, '犬': 50, '登记': 10, '狗': 50, '寄养': 50, '本市': 10, '户口': 10, '独户': 10, '居住': 10},
    {'港澳台居民定居证明签发': 100, '港澳台居民来大陆定居': 100, '台湾': 50, '定居': 50, '大陆': 30, '居民': 10, '申请': 10, '孤身': 10, '亲属': 10,
     '自理': 10},
    {'港澳台居民定居证明签发': 100, '前往港澳定居后返回内地定居': 100, '香港': 50, '澳门': 50, '定居': 50, '返回': 10, '内地': 30, '居民': 10, '放弃': 10,
     '身份': 10},
    {'大陆及台湾居民往来两地通行证及签注-台湾居民来往大陆通行证签发': 100, '台湾居民来往大陆通行证': 100, '台湾': 50, '居民': 30, '来往': 10, '大陆': 30, '内地': 30,
     '通行证': 50, '签发': 10, '申请': 10, '换发': 10,
     '补发': 10,
     '台胞证': 50, '委托': 10, '监护人': 10},
    {'大陆及台湾居民往来两地通行证及签注-大陆居民往来台湾通行证签发及签注': 100, '大陆居民往来台湾通行证': 100, '台湾': 50, '居民': 30, '往来': 10, '大陆': 30, '内地': 30,
     '通行证': 50, '签发': 10, '签注': 10, '旅游': 10},
    {'出入境通行证签发': 100, '出入境通行证': 100, '出入境': 50, '通行证': 50, '签发': 10, '申请人': 10, '国籍': 30, '冲突': 30, '不便': 10, '持用': 10,
     '普通': 10,
     '护照': 10, '港澳': 10, '香港': 10, '澳门': 10, '居民': 10, '内地': 10, '遗失': 10, '损毁': 10, '失效': 10, '返回': 10},
    {'外国人居留停留-外国人停留证件的签发、换发、补发': 100, '外国人停留证件': 100, '外国人': 30, '停留': 50, '证件': 10, '签发': 10, '换发': 10, '补发': 10,
     '免办': 10, '签证': 10, '入境': 10,
     '外交': 10, '公务': 10, '超过': 10, '免签期限': 10, '外国': 10, '船员': 10, '随行家属': 10, '离开': 10, '船舶': 10, '港口': 10, '退出': 10,
     '中国': 10, '国籍': 10, '境内': 10, '出生': 10, '婴儿': 10},
    {'外国人居留停留-外国人居留证件的签发、延期、换发、补发-团聚类居留证件签发、延期、换发、补发': 100, '外国人团聚类居留证件': 100, '外国人': 30, '团聚': 50, '居留': 50,
     '证件': 10, '签发': 10, '延期': 10, '换发': 10, '补发': 10, '申请': 10,
     '家庭': 10, '中国': 10, '境内': 10, '公民': 10, '成员': 10, '永久': 10, '资格': 10, '寄养': 10},
    {'外国人居留停留-外国人居留证件的签发、延期、换发、补发-学习类居留证件签发、延期、换发、补发': 100, '外国人学习类居留证件': 100, '外国人': 30, '学习': 50, '居留': 50,
     '证件': 10, '签发': 10, '延期': 10, '换发': 10, '补发': 10},
    {'外国人居留停留-外国人居留证件的签发、延期、换发、补发-工作类居留证件签发、延期、换发、补发': 100, '外国人工作类居留证件': 100, '外国人': 30, '工作': 50, '居留': 50,
     '证件': 10, '签发': 10, '延期': 10, '换发': 10, '补发': 10, '中国': 10,
     '境内': 10},
    {'外国人居留停留-外国人居留证件的签发、延期、换发、补发-记者类居留证件签发、延期、换发、补发': 100, '外国人记者类居留证件': 100, '外国人': 30, '记者': 50, '居留': 50,
     '证件': 10, '签发': 10, '延期': 10, '换发': 10, '补发': 10, '外国': 10,
     '新闻机构': 30, '常驻': 10, '中国': 10},
    {'内地居民前往港澳通行证、往来港澳通行证和签注核发': 100, '内地居民往来港澳通行证': 100, '内地': 30, '大陆': 30, '居民': 30, '往来': 10, '港澳': 40, '通行证': 50,
     '香港': 30, '澳门': 30, '前往': 10,
     '签注': 50,
     '核发': 10}
]

service = ['普通护照签发、换发、补发及加注--首次申请护照',
           '普通护照签发、换发、补发及加注--护照加注',
           '普通护照签发、换发、补发及加注--护照补发',
           '普通护照签发、换发、补发及加注--换发护照',
           '机动车登记与临时通行牌证核发--机动车登记',
           '机动车登记与临时通行牌证核发--机动车登记',
           '机动车登记与临时通行牌证核发--机动车登记',
           '机动车登记与临时通行牌证核发--机动车登记',
           '机动车登记与临时通行牌证核发--机动车登记',
           '机动车登记与临时通行牌证核发--机动车登记',
           '机动车登记与临时通行牌证核发--机动车登记',
           '机动车登记与临时通行牌证核发--机动车登记',
           '机动车登记与临时通行牌证核发--机动车登记',
           '机动车登记与临时通行牌证核发--机动车登记',
           '机动车登记与临时通行牌证核发--机动车登记',
           '机动车登记与临时通行牌证核发--机动车登记',
           '机动车登记与临时通行牌证核发--机动车登记',
           '机动车登记与临时通行牌证核发--机动车登记',
           '机动车登记与临时通行牌证核发--机动车登记',
           '客运出租汽车经营资格证件核发--客运出租汽车经营资格证核发',
           '客运出租汽车经营资格证件核发--网络预约出租汽车驾驶员证核发--网络预约出租汽车驾驶员证核发',
           '客运出租汽车经营资格证件核发--驾驶员客运资格证核发',
           '教师资格的认定', '养犬登记',
           '港澳台居民定居证明签发',
           '港澳台居民定居证明签发',
           '大陆及台湾居民往来两地通行证及签注--台湾居民来往大陆通行证签发',
           '大陆及台湾居民往来两地通行证及签注--大陆居民往来台湾通行证签发及签注',
           '出入境通行证签发',
           '外国人居留停留--外国人停留证件的签发、换发、补发',
           '外国人居留停留--外国人居留证件的签发、延期、换发、补发--团聚类居留证件签发、延期、换发、补发',
           '外国人居留停留--外国人居留证件的签发、延期、换发、补发--学习类居留证件签发、延期、换发、补发',
           '外国人居留停留--外国人居留证件的签发、延期、换发、补发--工作类居留证件签发、延期、换发、补发',
           '外国人居留停留--外国人居留证件的签发、延期、换发、补发--记者类居留证件签发、延期、换发、补发',
           '内地居民前往港澳通行证、往来港澳通行证和签注核发'
           ]

slot_max_weight = {'首次申请护照': 100, '护照加注': 100, '护照补发': 100, '换发护照': 100,
                   '初次申领机动车号牌和行驶证': 100, '改变车身颜色': 100,
                   '更换发动机': 100, '更换车身车架': 100, '因质量问题更换整车': 100, '改变运营模式和使用性质': 100, '车辆管理所管辖区域改变': 100,
                   '已注册登记车辆所有权发生改变': 100, '机动车抵押': 100, '已达到国家强制报废标准': 100, '灭失': 100, '因故不在我国境内使用': 100,
                   '因质量问题退车': 100, '机动车登记被依法撤销': 100, '被依法收缴并强制报废的': 100, '客运出租汽车经营资格证核发': 100,
                   '网络预约出租汽车驾驶员证核发': 100, '驾驶员客运资格证核发': 100, '认定教师资格': 100, '养犬登记': 100, '港澳台居民来大陆定居': 100,
                   '前往港澳定居后返回内地定居': 100, '台湾居民来往大陆通行证': 100, '大陆居民往来台湾通行证': 100, '出入境通行证': 100, '外国人停留证件': 100,
                   '外国人团聚类居留证件': 100, '外国人学习类居留证件': 100, '外国人工作类居留证件': 100, '外国人记者类居留证件': 100, '内地居民往来港澳通行证': 100
                   }
# slot_max = list(slot_max_weight.keys())

action_set = {'request': 0, 'inform': 1, 'closing': 2, 'deny': 3}
