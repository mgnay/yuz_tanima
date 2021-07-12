
gercek = table2array(sonuc(:,1));
tahmin = table2array(sonuc(:,2));

cm = confusionchart(gercek, tahmin);
cm.Title = "Yüz Tanıma Sınıflandırma Sonuçları";
%cm.RowSummary = 'row-normalized';
%cm.ColumnSummary = 'column-normalized';
sortClasses(cm,["Adile Nasit" "Ahmet Ariman" "Cem Gurdap" "Cengiz Nezir" "Feridun Savli" "Halit Akcatepe" "Kemal Sunal" "Munir Ozkul" "Sener Sen" "Tarik Akan" "Unknown"]);